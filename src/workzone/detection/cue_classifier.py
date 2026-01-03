"""
Cue Classifier - Maps YOLO detections to logical cue groups
Phase 1.1: Multi-Cue AND + Temporal Persistence
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np


@dataclass
class CueDetection:
    """Single cue detection with metadata."""
    cue_group: str
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    count: int = 1


@dataclass
class FrameCues:
    """Aggregated cues for a single frame."""
    frame_id: int
    timestamp: float
    cue_groups: Dict[str, Dict]  # {group_name: {present, count, max_conf, detections}}
    
    def __repr__(self):
        present_cues = [k for k, v in self.cue_groups.items() if v['present']]
        return f"Frame {self.frame_id}: {len(present_cues)} cues ({', '.join(present_cues)})"


class CueClassifier:
    """
    Maps YOLO class predictions to logical cue groups.
    
    Cue groups represent independent evidence types:
    - CHANNELIZATION: cones, barriers, delineators
    - SIGNAGE: work zone signs, message boards
    - PERSONNEL: workers, flaggers, PPE
    - EQUIPMENT: construction vehicles, machinery
    - INFRASTRUCTURE: temporary pavement, lane shifts
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize cue classifier with configuration.
        
        Args:
            config_path: Path to multi_cue_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "multi_cue_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Build class -> cue_group mapping
        self.class_to_cue = self._build_class_mapping()
        
        # Get thresholds
        self.detection_thresholds = self.config['detection_thresholds']
        self.min_objects_per_cue = self.config['min_objects_per_cue']
        
        # Get cue group names
        self.cue_groups = list(self.config['cue_groups'].keys())
        
    def _build_class_mapping(self) -> Dict[str, str]:
        """Build mapping from YOLO class name to cue group."""
        mapping = {}
        for group_name, group_info in self.config['cue_groups'].items():
            for class_name in group_info['classes']:
                mapping[class_name.lower()] = group_name
        return mapping
    
    def classify_detections(
        self,
        yolo_results,
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> FrameCues:
        """
        Classify YOLO detections into cue groups.
        
        Args:
            yolo_results: YOLO prediction results (ultralytics format)
            frame_id: Frame number
            timestamp: Video timestamp in seconds
            
        Returns:
            FrameCues object with aggregated cue information
        """
        # Initialize cue groups
        cue_data = {
            group: {
                'present': False,
                'count': 0,
                'max_conf': 0.0,
                'detections': []
            }
            for group in self.cue_groups
        }
        
        # Parse YOLO results
        if yolo_results is None or len(yolo_results) == 0:
            return FrameCues(frame_id, timestamp, cue_data)
        
        # Handle ultralytics Results object
        if hasattr(yolo_results, 'boxes'):
            boxes = yolo_results.boxes
            if boxes is None or len(boxes) == 0:
                return FrameCues(frame_id, timestamp, cue_data)
            
            # Extract detections
            for box in boxes:
                class_id = int(box.cls.cpu().item())
                confidence = float(box.conf.cpu().item())
                bbox = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
                
                # Get class name from YOLO model
                if hasattr(yolo_results, 'names'):
                    class_name = yolo_results.names[class_id].lower()
                else:
                    class_name = f"class_{class_id}"
                
                # Map to cue group
                cue_group = self.class_to_cue.get(class_name)
                if cue_group is None:
                    continue  # Skip unmapped classes
                
                # Check confidence threshold
                if confidence < self.detection_thresholds[cue_group]:
                    continue  # Below threshold, skip
                
                # Add detection
                detection = CueDetection(
                    cue_group=cue_group,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=tuple(bbox)
                )
                cue_data[cue_group]['detections'].append(detection)
                cue_data[cue_group]['count'] += 1
                cue_data[cue_group]['max_conf'] = max(
                    cue_data[cue_group]['max_conf'],
                    confidence
                )
        
        # Determine presence based on min_objects threshold
        for group in self.cue_groups:
            min_count = self.min_objects_per_cue[group]
            cue_data[group]['present'] = (cue_data[group]['count'] >= min_count)
        
        return FrameCues(frame_id, timestamp, cue_data)
    
    def get_cue_summary(self, frame_cues: FrameCues) -> Dict:
        """
        Get human-readable summary of frame cues.
        
        Returns:
            Dictionary with cue statistics
        """
        present_cues = [
            (group, data['count'], data['max_conf'])
            for group, data in frame_cues.cue_groups.items()
            if data['present']
        ]
        
        return {
            'frame_id': frame_cues.frame_id,
            'timestamp': frame_cues.timestamp,
            'num_cue_types': len(present_cues),
            'present_cues': present_cues,
            'total_detections': sum(
                data['count'] for data in frame_cues.cue_groups.values()
            )
        }
    
    def get_supported_classes(self) -> Dict[str, List[str]]:
        """Get all supported YOLO classes by cue group."""
        return {
            group: info['classes']
            for group, info in self.config['cue_groups'].items()
        }


# ===================================
# UTILITY FUNCTIONS
# ===================================

def print_cue_mapping(classifier: CueClassifier):
    """Print cue group mapping for debugging."""
    print("\n=== CUE CLASSIFICATION MAPPING ===")
    for group in classifier.cue_groups:
        classes = classifier.get_supported_classes()[group]
        print(f"\n{group} ({len(classes)} classes):")
        print(f"  Threshold: {classifier.detection_thresholds[group]}")
        print(f"  Min objects: {classifier.min_objects_per_cue[group]}")
        print(f"  Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")


if __name__ == "__main__":
    # Test cue classifier
    classifier = CueClassifier()
    print_cue_mapping(classifier)
    
    print(f"\n✓ Cue Classifier loaded successfully")
    print(f"  - {len(classifier.cue_groups)} cue groups")
    print(f"  - {len(classifier.class_to_cue)} mapped classes")
