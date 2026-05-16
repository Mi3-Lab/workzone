from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


StateIntervals = Dict[str, List[Tuple[int, int]]]


@dataclass
class VideoGroundTruth:
    states: StateIntervals


@dataclass
class VideoPredictions:
    states: Optional[StateIntervals]
    fps: Optional[float]
    detections: Optional[Any]
    ocr: Optional[Any]
