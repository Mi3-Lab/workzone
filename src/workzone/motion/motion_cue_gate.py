"""
Motion cue integration into Phase 1.1 multi-cue system.

Adds motion plausibility validation as a cue group alongside
CHANNELIZATION, SIGNAGE, PERSONNEL, EQUIPMENT, INFRASTRUCTURE.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from workzone.motion import MotionCueDetector, MotionMetrics


@dataclass
class MotionCueResult:
    """Result of motion cue evaluation."""
    is_plausible: bool
    confidence: float  # [0-1]
    motion_type: str
    avg_plausibility: float


class MotionCueGate:
    """Evaluate motion plausibility as a binary gate (0 or 1)."""
    
    def __init__(self,
                 plausibility_threshold: float = 0.4,
                 use_coherence_penalty: bool = True):
        """
        Initialize motion cue gate.
        
        Args:
            plausibility_threshold: Min plausibility to pass gate
            use_coherence_penalty: Apply additional coherence penalty
        """
        self.plausibility_threshold = plausibility_threshold
        self.use_coherence_penalty = use_coherence_penalty
        self.motion_detector = MotionCueDetector(method="lk")
    
    def evaluate(self, frame: np.ndarray,
                 detections: List[Dict]) -> MotionCueResult:
        """
        Evaluate motion cues from frame detections.
        
        Args:
            frame: Input frame (BGR)
            detections: YOLO detection results
        
        Returns:
            MotionCueResult with plausibility assessment
        """
        # Convert YOLO results to detection dicts
        detection_dicts = self._convert_yolo_detections(detections)
        
        # Process with motion detector
        self.motion_detector.process_frame(frame, detection_dicts)
        
        # Evaluate motion plausibility
        if not detection_dicts:
            return MotionCueResult(
                is_plausible=True,  # No detections = pass
                confidence=1.0,
                motion_type="no_detections",
                avg_plausibility=1.0
            )
        
        # Average plausibility across detections
        plausibilities = [
            det['motion_metrics'].plausibility
            for det in detection_dicts
            if 'motion_metrics' in det
        ]
        
        if not plausibilities:
            return MotionCueResult(
                is_plausible=True,
                confidence=1.0,
                motion_type="no_motion_metrics",
                avg_plausibility=1.0
            )
        
        avg_plausibility = float(np.mean(plausibilities))
        min_plausibility = float(np.min(plausibilities))
        
        # Apply coherence penalty if enabled
        if self.use_coherence_penalty:
            coherences = [
                det['motion_metrics'].motion_coherence
                for det in detection_dicts
                if 'motion_metrics' in det
            ]
            avg_coherence = float(np.mean(coherences)) if coherences else 0.5
            penalized = avg_plausibility * (0.8 + 0.2 * avg_coherence)
        else:
            penalized = avg_plausibility
        
        is_plausible = penalized >= self.plausibility_threshold
        
        # Motion type: majority class
        motion_types = [
            det['motion_metrics'].motion_type
            for det in detection_dicts
            if 'motion_metrics' in det
        ]
        motion_type = max(set(motion_types), key=motion_types.count) if motion_types else "unknown"
        
        return MotionCueResult(
            is_plausible=is_plausible,
            confidence=float(penalized),
            motion_type=motion_type,
            avg_plausibility=avg_plausibility
        )
    
    def _convert_yolo_detections(self, yolo_results) -> List[Dict]:
        """Convert YOLO Results object to detection dicts."""
        detections = []
        
        if not hasattr(yolo_results, 'boxes') or yolo_results.boxes is None:
            return detections
        
        boxes = yolo_results.boxes
        names = yolo_results.names
        
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            
            detections.append({
                'bbox': bbox,
                'class_id': cls_id,
                'class_name': names[cls_id],
                'confidence': conf
            })
        
        return detections
