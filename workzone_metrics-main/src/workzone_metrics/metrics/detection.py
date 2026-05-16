from typing import Any, Dict


def compute_map_50(detections: Any, ground_truth: Any) -> Dict[str, float]:
    raise NotImplementedError(
        "mAP@0.5 requires per-frame bounding boxes in a supported format. "
        "Provide detections and ground-truth boxes in COCO-like format, then implement this function."
    )


def compute_precision_at_high_recall(detections: Any, ground_truth: Any, recall_target: float = 0.9) -> float:
    raise NotImplementedError(
        "Precision@high-recall requires scored detections with IoU matching. "
        "Provide detections and ground-truth boxes in COCO-like format, then implement this function."
    )
