# Phase 1.3: Motion/Flow Cues for Work Zone Detection

**Goal**: Use optical flow and motion validation to distinguish real construction equipment from dynamic false positives (flapping tarps, swaying vegetation, distant traffic).

**Estimated Impact**: 20-30% additional false positive reduction through motion plausibility filtering.

## Overview

Phase 1.3 adds a new dimension orthogonal to appearance-based detection:

```
Phase 1.1 (Appearance):
  - YOLO detection of 50 classes
  - Multi-cue AND logic (CHANNELIZATION, SIGNAGE, PERSONNEL, EQUIPMENT, INFRASTRUCTURE)
  - Temporal persistence tracking

Phase 1.3 (Motion):
  + Optical flow estimation (Lucas-Kanade or Farneback)
  + Motion coherence analysis
  + Class-specific motion expectations (static vs dynamic)
  + Ego-motion estimation
  + Physical plausibility scoring
  
Result: Appearance + Motion = More robust false positive rejection
```

## Key Concepts

### 1. Optical Flow

**What it does**: Estimates pixel-level motion between consecutive frames.

**Methods**:
- **Lucas-Kanade (LK)**: Sparse flow, ~2-3ms, good for real-time
- **Farneback**: Dense flow, ~5-10ms, more robust but slower
- **Frame Differencing**: Trivial, <1ms, motion magnitude only

**Why it helps**:
- Catches **dynamic false positives**: flapping tarps, flags, vegetation
- Validates **static object motion**: cones should move with camera, not independently
- Independent from color/shape: catches failures of appearance-only models

### 2. Motion Metrics

For each detection, compute:

```
box_motion_magnitude   # Average motion within bounding box (pixels)
ego_motion_estimate    # Estimated camera motion (from background)
motion_coherence       # How uniform is motion (0=scattered, 1=uniform)
plausibility           # Physics-based score (0-1)
motion_type            # "static", "dynamic", "inconsistent", "incoherent"
```

### 3. Class-Specific Expectations

**Static Classes** (Cones, Barriers, Signs):
- Should move ≈ ego-motion (camera motion)
- If moving independently → likely false positive
- Plausibility ∝ coherence of motion

**Dynamic Classes** (Workers, Vehicles):
- Can move independently
- If moving coherently → good sign
- If incoherent motion → suspicious

**Semi-Dynamic** (Infrastructure):
- Can be either, depends on context

### 4. Physical Plausibility

Plausibility score reflects whether detected object's motion is physically reasonable:

```python
If static_class:
    if box_motion < static_threshold:
        plausibility = 0.8 + 0.2 * coherence  # Good: cone is static
    elif box_motion < dynamic_threshold:
        plausibility = 0.3  # Suspicious: cone is moving
    else:
        plausibility = 0.1  # Bad: cone is wildly moving

If dynamic_class:
    if box_motion < static_threshold:
        plausibility = 0.4  # Suspicious: worker not moving
    elif box_motion < dynamic_threshold:
        plausibility = 0.6 + 0.2 * coherence  # OK: moderate motion
    else:
        plausibility = 0.7 + 0.2 * coherence  # Good: strong coherent motion
```

## Integration with Phase 1.1

Two integration modes:

### Mode 1: Hard Gate (require_motion_plausibility=true)

Work zone detection requires BOTH:
- Phase 1.1: Multi-cue logic passes (≥1 sustained cue type)
- **AND** Phase 1.3: Motion validation passes (plausibility ≥ 0.40)

```python
if multi_cue_pass and motion_pass:
    state = INSIDE
```

**Pros**: Eliminates dynamic false positives  
**Cons**: May miss real work zones with unusual motion

### Mode 2: Soft Penalty (require_motion_plausibility=false)

Phase 1.3 applies penalty to plausibility score:

```python
adjusted_plausibility = multi_cue_confidence * motion_penalty
motion_penalty = plausibility_penalty_factor if motion_fails else 1.0
```

**Pros**: Soft feedback, less likely to miss real zones  
**Cons**: May not fully eliminate false positives

## Configuration

Edit `configs/motion_cue_config.yaml`:

```yaml
optical_flow:
  method: "lk"  # or "farneback"
  pyramid_levels: 2

motion_validation:
  static_threshold: 0.15  # pixels, max for static
  dynamic_threshold: 5.0  # pixels, min for clearly moving
  coherence_threshold: 0.4  # [0-1]
  
motion_cue_gate:
  plausibility_threshold: 0.40  # Pass if ≥ this
  use_coherence_penalty: true

phase1_1_integration:
  enabled: true
  require_motion_plausibility: false  # Use soft penalty
  plausibility_penalty_factor: 0.8
```

## Example Usage

### In Video Processing Script

```python
from workzone.motion import MotionCueGate
from ultralytics import YOLO

yolo = YOLO("weights/yolo12s_hardneg_1280.pt")
motion_gate = MotionCueGate(plausibility_threshold=0.40)

for frame in video:
    results = yolo(frame)
    
    # Evaluate motion
    motion_result = motion_gate.evaluate(frame, results)
    
    if motion_result.is_plausible:
        # Proceed with normal Phase 1.1 logic
        print(f"Motion OK: {motion_result.motion_type}")
    else:
        # Penalize or reject detection
        print(f"Implausible motion: {motion_result.avg_plausibility:.2f}")
```

### Visualize Motion

```python
from workzone.motion import MotionCueDetector

detector = MotionCueDetector(method="lk", enable_visualization=True)

for frame in video:
    results = yolo(frame)
    detector.process_frame(frame, detections)
    
    # Draw flow vectors and plausibility boxes
    vis = detector.draw_flow_visualization(frame, flow, detections)
    cv2.imshow("Motion Analysis", vis)
```

## Performance

**Optical Flow Latency**:
- Lucas-Kanade: 2-3ms (200 features, 15×15 window)
- Farneback: 5-10ms (dense 3-level pyramid)
- Frame Diff: <1ms (trivial)

**Total Pipeline Latency** (with Phase 1.1):
- Base YOLO: 12ms
- + Phase 1.1: <1ms
- + Phase 1.3 (LK): 2-3ms
- **Total: 14-15ms → ~65-70 FPS** ✓ Real-time on A100

**Memory**:
- Optical flow: ~H×W×2×4 bytes (for 1280×720 = 7.4 MB)
- No additional GPU memory needed

## Testing Strategy


### 2. False Positive Analysis

Test on hard negatives with dynamic elements:
- Flapping orange tarps
- Swaying traffic signs
- Distant vehicles (orange trucks)
- Road reflections

Expected: Phase 1.3 rejects 60-80% of these.

### 3. Real Work Zone Validation

Ensure real work zones still pass:
- Static cones with camera motion
- Workers moving in structured way
- Construction equipment idling

Expected: Phase 1.3 doesn't significantly increase false negatives.

## Roadmap (Future Improvements)

### Short Term (Phase 1.3.1)
- [ ] Trajectory-based validation (track boxes over time)
- [ ] Kalman filtering for smoother motion estimates
- [ ] Per-class motion history (learn typical patterns)

### Medium Term (Phase 1.3.2)
- [ ] Dense temporal segmentation (what part of image belongs to object?)
- [ ] Ego-motion subtraction for cleaner motion analysis
- [ ] Background/foreground separation (separate object motion from scene)

### Long Term (Phase 2.0)
- [ ] Recurrent neural networks for motion understanding
- [ ] Joint YOLO + optical flow training
- [ ] End-to-end learned plausibility scoring

## References

- Lucas-Kanade optical flow: Lucas & Kanade (1981)
- Farneback dense flow: Farneback (2003)
- OpenCV optical flow: https://docs.opencv.org/4.x/d7/d8b/tutorial_py_lucas_kanade.html

---

**Status**: Phase 1.3 Implementation In Progress  
**Target Completion**: January 2026  
**Expected FP Reduction**: 84.6% (Phase 1.2) + 20-30% (Phase 1.3) = ~90%
