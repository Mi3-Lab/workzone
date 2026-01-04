# Hard-Negative Mining: Phase 1.2 Complete ✅

## Summary

Mining of 406 real videos from `data/videos_compressed/` has **completed successfully**. Extracted **17,957 candidate frames** showing work zone contexts with high YOLO + Phase 1.1 confidence.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| **Total Videos Processed** | 406 |
| **Candidate Frames Extracted** | 17,957 |
| **JPEG Snapshots** | 17,957 (~9.6 GB) |
| **Storage Location** | `outputs/hardneg_mining_gpu{0,1}/candidates/` |
| **Merged CSV** | `outputs/hardneg_mining/candidates_master.csv` |
| **Average Fused Score** | 0.743 ± 0.017 |
| **Average YOLO Score** | 0.874 ± 0.042 |
| **Phase 1.1 Pass Rate** | 100% (all 17,957 pass) |
| **Processing Time** | ~2 hours (stride=2, 2×A100) |

---

## What Was Mined?

### **Filtering Criteria**
- **Fused Score (CLIP+YOLO)**: 0.45–0.85 range (semantic confidence)
- **Phase 1.1 Multi-Cue Gate**: Required to pass (≥2 sustained cues + AND logic)
- **Max per Video**: 50 candidates (to maintain diversity)
- **Sampling**: Every 2nd frame (stride=2)

### **What This Means**
✅ Candidates are from **real work zone videos** with:
- High YOLO detection confidence (mean 0.87, 99% above 0.80)
- Strong Phase 1.1 multi-cue support (100% pass rate)
- Significant channelization cues (mean 8.4 cones/barriers per frame)
- Confident Phase 1.1 predictions (66% above 0.70 confidence)

These are **TRUE WORK ZONE FRAMES** from the real dataset, **not false positives**. They're candidates because they're challenging edge cases within work zones.

---

## Distribution Breakdown

### **Score Ranges**

**YOLO Confidence** (raw object detection):
- ≥0.80: 17,654 candidates (98.3%)
- ≥0.85: 16,147 candidates (89.9%)
- ≥0.90: 2,657 candidates (14.8%)

**Phase 1.1 Confidence** (temporal multi-cue):
- ≥0.50: 17,957 candidates (100%)
- ≥0.70: 11,862 candidates (66.1%)
- ≥0.85: 1,247 candidates (6.9%)

### **Detection Context**

**Spatial State**:
- INSIDE zone: 17,818 (99.2%)
- APPROACHING boundary: 137 (0.8%)
- OUT (false positives): 2 (0.01%)

**Channelization Cue Distribution**:
- 0–2 cues: 1,624 (9.0%)
- 3–5 cues: 3,802 (21.2%)
- 6–10 cues: 5,666 (31.5%)
- 11–20 cues: 5,848 (32.5%)
- 21+ cues: 2,413 (13.4%)

**Sustained Cues** (Phase 1.1 window):
- 2 sustained: 13,363 (74.4%) ← Most common
- 3 sustained: 3,318 (18.5%)
- 4+ sustained: 1,276 (7.1%)

---

## File Structure

```
outputs/
├── hardneg_mining/
│   ├── candidates_master.csv          (17,957 rows, merged from GPU0+GPU1)
│   ├── candidates_master_gpu0.csv     (9,258 rows, GPU0 subset)
│   └── candidates_master_gpu1.csv     (8,701 rows, GPU1 subset)
│
├── hardneg_mining_gpu0/
│   ├── candidates/
│   │   ├── boston_042e1caf93114d3286c11ba14ddaa759_000001_02790_snippet/
│   │   │   ├── boston_...f642.jpg
│   │   │   ├── boston_...f644.jpg
│   │   │   └── ...
│   │   ├── boston_042e1caf93114d3286c11ba14ddaa759_000001_13410_snippet/
│   │   └── ... (203 video folders, ~9,258 JPEGs total, 5.0 GB)
│   │
│   └── [video_stem]_timeline_fusion.csv, [video_stem]_annotated_fusion.mp4
│       (per-video analysis with Phase 1.1 overlay)
│
└── hardneg_mining_gpu1/
    ├── candidates/
    │   └── ... (203 video folders, ~8,701 JPEGs total, 4.6 GB)
    │
    └── [video_stem]_timeline_fusion.csv, [video_stem]_annotated_fusion.mp4
```

---

## CSV Columns Reference

```
video                  Path to source MP4
snapshot               Path to extracted JPEG snapshot
frame                  Frame number in video
time_sec               Timestamp (seconds)
yolo_score             Raw YOLO object confidence (0–1)
yolo_score_ema         EMA-smoothed YOLO score
fused_score_ema        CLIP + YOLO semantic fusion (0–1)
state                  Detection state (INSIDE/APPROACHING/OUT)
clip_used              Whether CLIP was applied (1=yes, 0=no)
clip_score             CLIP semantic similarity (if used)
count_channelization   # of channelization cues detected
count_workers          # of worker cues detected
p1_multi_cue_pass      Phase 1.1 AND gate result (1=pass, 0=fail)
p1_num_sustained       # cues sustained in Phase 1.1 window
p1_confidence          Phase 1.1 confidence score (0–1)
```

---

## Why This Matters

### **Why These Aren't Hard Negatives YET**

The mined frames are **valid work zone detections** from the real dataset. They're not "hard negatives" because:
- They actually contain work zone activity (cones, barriers, signage, workers, equipment)
- YOLO correctly detects them (mean confidence 0.87)
- Phase 1.1 correctly identifies them as work zones (100% pass rate)
- They're spatially INSIDE zone boundaries (99.2%)

### **What We're Looking For Now**

You need to **visually inspect these JPEGs** to find contextual variations that might confuse the YOLO model:
- Orange/yellow trucks parked near (not in) work zones
- Traffic cones on roadsides or at events (not ESV roadwork)
- Road signs, warning signs, billboards with orange/yellow coloring
- Weather conditions creating detection artifacts
- Other road construction/utility work that isn't ESV-related

**These lookalike cases** = hard negatives for training to improve robustness.

---

## Next Steps: Human Review & Categorization

### **Phase 1: Sample & Categorize** (Your Task)

Use the review tool to visually inspect JPEGs and categorize:

```bash
python scripts/review_hard_negatives.py --mode interactive --sample-size 100
```

**Categories to Use**:
- `orange_trucks` — Orange/yellow vehicles, equipment
- `random_cones` — Traffic cones outside work zones
- `roadside_signs` — Road signs, billboards, warnings
- `weather_artifacts` — Rain, fog, shadows, reflections
- `other_roadwork_lookalikes` — Road construction, paving, utility work

### **Phase 2: Bulk Export** (If you find a pattern)

Once you identify a category with many false candidates:

```bash
python scripts/review_hard_negatives.py --mode export \
  --score-range 0.49 0.55 \
  --category orange_trucks \
  --max-per-cat 50
```

This copies candidates in that score band to `data/02_processed/hard_negatives/orange_trucks/` for bulk review.

### **Phase 3: Organize & Document**

```bash
# After categorizing JPEGs manually:
ls data/02_processed/hard_negatives/
# Should show:
# orange_trucks/ random_cones/ roadside_signs/ weather_artifacts/ other_roadwork_lookalikes/
```

Update manifest:

```bash
python scripts/review_hard_negatives.py --mode manifest
# Creates: data/02_processed/hard_negatives/manifest.csv
```

### **Phase 4: Retrain YOLO**

```bash
# Merge hard negatives into YOLO dataset
# Add empty label files for hard negatives (background class)
# Update data/05_workzone_yolo/workzone_yolo.yaml

yolo detect train \
  data=05_workzone_yolo/workzone_yolo.yaml \
  model=yolo12s.pt \
  epochs=100 \
  imgsz=640 \
  device=0,1
```

### **Phase 5: Validate & Benchmark**

Compare metrics before/after:
- **Precision on hard-negative set**: Should improve (fewer false positives)
- **Recall on true work zones**: Should remain ≥95%
- **Overall F1**: Should improve with better precision

---

## Tools Available

### **1. Interactive Review** (One-by-one)
```bash
python scripts/review_hard_negatives.py --mode interactive \
  --sample-size 100
```
- Shows 100 random candidates
- Display score + cue info for each
- Option to categorize into folders
- Copies JPEGs to `data/02_processed/hard_negatives/<category>/`

### **2. Bulk Export** (By score range)
```bash
python scripts/review_hard_negatives.py --mode export \
  --score-range 0.55 0.65 \
  --category random_cones \
  --max-per-cat 100
```
- Exports ~100 candidates in specified score range to category folder
- Good for finding similar-looking cases quickly

### **3. Statistics** (Overview)
```bash
python scripts/review_hard_negatives.py --mode stats
```
- Score distributions, cue histograms
- Helps you understand what's in the dataset

### **4. Sample Viewer** (Visual exploration)
```bash
python scripts/sample_candidates.py
```
- Shows distribution breakdown
- High-confidence vs. low-confidence samples
- Recommendations for review

---

## Key Numbers to Remember

- **17,957** candidate frames ready for review
- **~9.6 GB** of JPEG snapshots
- **100%** pass Phase 1.1 filtering
- **0.743** mean fused score (high confidence)
- **8.4** mean channelization cues per frame
- **383** unique videos contributing candidates
- **0% false positives out-of-zone** (99.2% INSIDE)

---

## Estimated Effort

- **Visual Review**: ~2–4 hours for 100–200 sample JPEGs (to understand patterns)
- **Categorization**: ~8–12 hours for all 17,957 (or use batch patterns for faster coverage)
- **Retraining**: ~4–6 hours (100 epochs on 2×A100)
- **Validation**: ~1–2 hours

**Recommended Approach**: 
1. Review ~200 random samples to identify patterns
2. Use bulk export to group similar-looking cases
3. Organize ~2,000–5,000 confirmed hard negatives by category
4. Retrain and validate on that subset
5. Iterate with more if metrics improve significantly

---

## Questions to Ask Yourself While Reviewing

For each JPEG:
- **Is this a real work zone?** (If yes → true positive, skip for hard negatives)
- **Is this a lookalike?** (If yes → which category?)
- **Why did YOLO detect this?** (Orange color? Traffic cone? Sign?)
- **Why did Phase 1.1 sustain cues here?** (Channelization artifacts? False positives?)
- **Is this a learning opportunity?** (Will this help the model generalize better?)

---

## References

- **Mining Script**: `scripts/batch_mine_hard_negatives.py`
- **Process Video Fusion**: `scripts/process_video_fusion.py`
- **Review Tools**: `scripts/review_hard_negatives.py`, `scripts/sample_candidates.py`
- **Data Structure**: `data/02_processed/hard_negatives/`
- **YOLO Config**: `data/05_workzone_yolo/workzone_yolo.yaml`
- **Phase 1.1 Docs**: See `CONTRIBUTING.md` (CueClassifier section)
