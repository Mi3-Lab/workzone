from typing import Any


def compute_ocr_sign_accuracy(pred_ocr: Any, gt_ocr: Any) -> float:
    raise NotImplementedError(
        "OCR sign accuracy requires per-sign text annotations aligned to detections or frames."
    )
