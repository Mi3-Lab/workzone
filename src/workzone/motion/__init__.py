"""
Phase 1.3: Motion/Flow Cues for Work Zone Detection

Optical flow and motion analysis to:
- Distinguish real construction equipment from false positives
- Detect physical plausibility of detected objects
- Catch dynamic false positives (flapping tarps, swaying vegetation)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MotionMetrics:
    """Metrics describing motion of a bounding box region."""
    box_motion: np.ndarray  # (dx, dy) average motion of box contents
    box_motion_magnitude: float  # magnitude of box motion
    ego_motion_estimate: np.ndarray  # (dx, dy) estimated camera ego-motion
    ego_motion_magnitude: float  # magnitude of estimated ego-motion
    motion_coherence: float  # [0-1] how coherent is motion in box (1 = all pixels move same)
    plausibility: float  # [0-1] physical plausibility score
    motion_type: str  # "static", "dynamic", "inconsistent", "untrackable"


class OpticalFlowEstimator:
    """Lightweight optical flow estimation using Lucas-Kanade or frame differencing."""
    
    def __init__(self, method: str = "lk", pyramid_levels: int = 2):
        """
        Initialize optical flow estimator.
        
        Args:
            method: "lk" (Lucas-Kanade), "farneback" (Farneback), "diff" (frame diff)
            pyramid_levels: For Lucas-Kanade, number of pyramid levels
        """
        self.method = method
        self.pyramid_levels = pyramid_levels
        self.prev_gray = None
        self.prev_frame = None
        
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate optical flow from previous frame to current frame.
        
        Returns:
            flow: shape (H, W, 2) with (dx, dy) at each pixel
            or None if not enough frames yet
        """
        if frame is None or frame.size == 0:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame - initialize and return None
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame
            return None
        
        if self.method == "lk":
            return self._lucas_kanade(gray)
        elif self.method == "farneback":
            return self._farneback(gray)
        else:
            return self._frame_diff(gray)
    
    def _lucas_kanade(self, gray: np.ndarray) -> np.ndarray:
        """Sparse Lucas-Kanade optical flow."""
        # Detect good features to track
        corners = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
        
        if corners is None or len(corners) < 10:
            # Fallback to frame differencing
            return self._frame_diff(gray)
        
        # Calculate optical flow using Lucas-Kanade
        flow, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, corners, None,
            winSize=(15, 15),
            maxLevel=self.pyramid_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Interpolate sparse flow to dense
        if flow is not None and status is not None:
            valid_flow = flow[status.flatten() == 1]
            if len(valid_flow) > 0:
                # Create dense flow from sparse points
                h, w = gray.shape
                dense_flow = self._sparse_to_dense(corners, flow, status, (h, w))
                self.prev_gray = gray
                return dense_flow
        
        return self._frame_diff(gray)
    
    def _farneback(self, gray: np.ndarray) -> np.ndarray:
        """Dense Farneback optical flow."""
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            n8=True,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        self.prev_gray = gray
        return flow
    
    def _frame_diff(self, gray: np.ndarray) -> np.ndarray:
        """Simple frame differencing for motion magnitude."""
        h, w = gray.shape
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        
        # Create flow-like output (zero flow where no motion)
        flow = np.zeros((h, w, 2), dtype=np.float32)
        return flow
    
    def _sparse_to_dense(self, corners: np.ndarray, flow: np.ndarray, 
                         status: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Interpolate sparse flow to dense grid."""
        h, w = shape
        dense_flow = np.zeros((h, w, 2), dtype=np.float32)
        
        valid_idx = status.flatten() == 1
        if np.sum(valid_idx) > 0:
            pts = corners[valid_idx].reshape(-1, 2)
            flows = flow[valid_idx].reshape(-1, 2)
            
            # Simple nearest-neighbor interpolation for speed
            for (x, y), (dx, dy) in zip(pts, flows):
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    dense_flow[y, x] = [dx, dy]
        
        return dense_flow


class MotionValidator:
    """Validate motion plausibility of detected objects."""
    
    def __init__(self,
                 static_threshold: float = 0.15,  # pixels of motion acceptable for static
                 dynamic_threshold: float = 5.0,  # min pixels for moving object
                 coherence_threshold: float = 0.4,  # min coherence for valid motion
                 ego_motion_smooth: float = 0.7):  # EMA alpha for ego-motion
        """
        Initialize motion validator.
        
        Args:
            static_threshold: Max pixel motion for object to be considered "static"
            dynamic_threshold: Min pixel motion for object to be "dynamic"
            coherence_threshold: Min motion coherence for physical plausibility
            ego_motion_smooth: EMA smoothing for estimated ego-motion
        """
        self.static_threshold = static_threshold
        self.dynamic_threshold = dynamic_threshold
        self.coherence_threshold = coherence_threshold
        self.ego_motion_smooth = ego_motion_smooth
        self.ego_motion = np.array([0.0, 0.0], dtype=np.float32)
        
    def validate(self, flow: np.ndarray, bbox: Tuple[int, int, int, int],
                 class_name: str) -> MotionMetrics:
        """
        Validate motion plausibility of a bounding box.
        
        Args:
            flow: Optical flow array (H, W, 2)
            bbox: (x1, y1, x2, y2) bounding box coordinates
            class_name: Detection class name (e.g., "Cone", "Worker")
        
        Returns:
            MotionMetrics with plausibility score
        """
        if flow is None or flow.size == 0:
            return MotionMetrics(
                box_motion=np.array([0.0, 0.0]),
                box_motion_magnitude=0.0,
                ego_motion_estimate=self.ego_motion.copy(),
                ego_motion_magnitude=float(np.linalg.norm(self.ego_motion)),
                motion_coherence=0.5,
                plausibility=0.5,  # Unknown
                motion_type="untrackable"
            )
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = flow.shape[:2]
        
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return MotionMetrics(
                box_motion=np.array([0.0, 0.0]),
                box_motion_magnitude=0.0,
                ego_motion_estimate=self.ego_motion.copy(),
                ego_motion_magnitude=float(np.linalg.norm(self.ego_motion)),
                motion_coherence=0.0,
                plausibility=0.0,
                motion_type="invalid"
            )
        
        # Extract flow in bounding box
        box_flow = flow[y1:y2, x1:x2]
        
        if box_flow.size == 0:
            return MotionMetrics(
                box_motion=np.array([0.0, 0.0]),
                box_motion_magnitude=0.0,
                ego_motion_estimate=self.ego_motion.copy(),
                ego_motion_magnitude=float(np.linalg.norm(self.ego_motion)),
                motion_coherence=0.0,
                plausibility=0.0,
                motion_type="empty_box"
            )
        
        # Compute motion statistics
        avg_flow = np.mean(box_flow, axis=(0, 1))  # (dx, dy)
        motion_magnitude = float(np.linalg.norm(avg_flow))
        
        # Motion coherence: how consistent is flow direction
        magnitude_per_pixel = np.linalg.norm(box_flow, axis=2)
        if np.max(magnitude_per_pixel) > 0:
            normalized = magnitude_per_pixel / (np.max(magnitude_per_pixel) + 1e-6)
            motion_coherence = float(np.mean(normalized))
        else:
            motion_coherence = 0.0
        
        # Update ego-motion estimate (from background)
        self._update_ego_motion(flow)
        ego_magnitude = float(np.linalg.norm(self.ego_motion))
        
        # Determine motion type and plausibility
        motion_type, plausibility = self._classify_motion(
            motion_magnitude, ego_magnitude, motion_coherence, class_name
        )
        
        return MotionMetrics(
            box_motion=avg_flow.astype(np.float32),
            box_motion_magnitude=motion_magnitude,
            ego_motion_estimate=self.ego_motion.copy(),
            ego_motion_magnitude=ego_magnitude,
            motion_coherence=motion_coherence,
            plausibility=plausibility,
            motion_type=motion_type
        )
    
    def _update_ego_motion(self, flow: np.ndarray) -> None:
        """Estimate ego-motion from optical flow (median approach)."""
        if flow.size == 0:
            return
        
        magnitudes = np.linalg.norm(flow, axis=2)
        
        # Take low-motion region as ego-motion estimate
        # (background typically moves most consistently)
        threshold = np.percentile(magnitudes, 25)
        mask = magnitudes < (threshold + 0.5)
        
        if np.sum(mask) > 100:
            background_flow = flow[mask]
            new_ego = np.median(background_flow, axis=0)
        else:
            new_ego = np.array([0.0, 0.0], dtype=np.float32)
        
        # EMA smoothing
        self.ego_motion = (self.ego_motion_smooth * self.ego_motion + 
                          (1 - self.ego_motion_smooth) * new_ego)
    
    def _classify_motion(self, box_motion: float, ego_motion: float,
                         coherence: float, class_name: str) -> Tuple[str, float]:
        """
        Classify motion type and compute plausibility.
        
        Returns:
            (motion_type, plausibility_score)
        """
        # Expected motion types for classes
        static_classes = {"Cone", "Drum", "Barricade", "Barrier", "Fence",
                         "Tubular Marker", "Vertical Panel"}
        dynamic_classes = {"Worker", "Police Officer", "Work Vehicle", 
                          "Work Equipment", "Police Vehicle"}
        
        # Normalize motion by ego-motion
        if ego_motion > 0.1:
            relative_motion = box_motion / ego_motion
        else:
            relative_motion = box_motion
        
        is_static_class = class_name in static_classes
        is_dynamic_class = class_name in dynamic_classes
        
        if box_motion < self.static_threshold:
            motion_type = "static"
            # Static objects should have low motion
            if is_static_class:
                plausibility = 0.8 + (0.2 * coherence)  # Good: cone is static
            else:
                plausibility = 0.4  # Suspicious: dynamic class is static
        
        elif box_motion < self.dynamic_threshold:
            motion_type = "inconsistent"
            # Moderate motion - could be either
            if is_static_class:
                plausibility = 0.3  # Bad: cone is moving
            else:
                plausibility = 0.6 + (0.2 * coherence)  # OK: worker might move slightly
        
        else:
            motion_type = "dynamic"
            # Strong motion
            if is_dynamic_class:
                # Dynamic class moving - good if coherent
                plausibility = 0.7 + (0.2 * coherence)
            else:
                # Static class moving - very bad
                plausibility = 0.1
        
        # Coherence penalty for incoherent motion
        if coherence < self.coherence_threshold:
            plausibility *= 0.6
            motion_type = "incoherent"
        
        return motion_type, min(1.0, max(0.0, plausibility))


class MotionCueDetector:
    """Main detector combining optical flow and motion validation."""
    
    def __init__(self, method: str = "lk", enable_visualization: bool = False):
        """Initialize motion cue detector."""
        self.flow_estimator = OpticalFlowEstimator(method=method)
        self.validator = MotionValidator()
        self.enable_visualization = enable_visualization
        
    def process_frame(self, frame: np.ndarray,
                      detections: List[Dict]) -> List[Dict]:
        """
        Add motion metrics to detections.
        
        Args:
            frame: Input frame (BGR)
            detections: List of detection dicts with 'bbox', 'class', 'conf'
        
        Returns:
            Detections with added 'motion_metrics' field
        """
        # Estimate optical flow
        flow = self.flow_estimator.estimate(frame)
        
        # Validate motion for each detection
        for det in detections:
            if flow is not None:
                metrics = self.validator.validate(
                    flow,
                    det['bbox'],
                    det.get('class_name', 'Unknown')
                )
            else:
                # No flow yet
                metrics = MotionMetrics(
                    box_motion=np.array([0.0, 0.0]),
                    box_motion_magnitude=0.0,
                    ego_motion_estimate=np.array([0.0, 0.0]),
                    ego_motion_magnitude=0.0,
                    motion_coherence=0.5,
                    plausibility=0.5,
                    motion_type="warmup"
                )
            
            det['motion_metrics'] = metrics
        
        return detections
    
    def draw_flow_visualization(self, frame: np.ndarray, flow: Optional[np.ndarray],
                                detections: List[Dict]) -> np.ndarray:
        """Draw optical flow and motion metrics on frame."""
        if flow is None or flow.size == 0:
            return frame
        
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Draw optical flow vectors
        step = 15  # Visualization step
        for y in range(0, h, step):
            for x in range(0, w, step):
                if x < w and y < h:
                    fx, fy = flow[y, x]
                    x2, y2 = int(x + fx), int(y + fy)
                    if 0 <= x2 < w and 0 <= y2 < h:
                        cv2.arrowedLine(vis, (x, y), (x2, y2),
                                       (0, 255, 0), 1, tipLength=0.2)
        
        # Draw detection motion info
        for det in detections:
            if 'motion_metrics' not in det:
                continue
            
            metrics = det['motion_metrics']
            bbox = [int(v) for v in det['bbox']]
            
            # Color by plausibility
            if metrics.plausibility > 0.7:
                color = (0, 255, 0)  # Green - plausible
            elif metrics.plausibility > 0.4:
                color = (0, 165, 255)  # Orange - uncertain
            else:
                color = (0, 0, 255)  # Red - implausible
            
            cv2.rectangle(vis, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            
            # Label with motion type and plausibility
            label = f"{metrics.motion_type[:4]} {metrics.plausibility:.2f}"
            cv2.putText(vis, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis


# Import MotionCueGate for convenient access
from .motion_cue_gate import MotionCueGate, MotionCueResult

__all__ = [
    'OpticalFlowEstimator',
    'MotionValidator', 
    'MotionMetrics',
    'MotionCueDetector',
    'MotionCueGate',
    'MotionCueResult'
]
