#!/usr/bin/env python3
"""
Minimal example of integrating text_confidence into state machine.

This shows how to modify the fused score calculation to include OCR confidence.
"""

# Current scoring (line 534-544 in app_phase2_1_evaluation.py):
def scoring_example_before():
    """Original scoring without OCR"""
    yolo_score = 0.7
    clip_enabled = True
    clip_weight = 0.3
    clip_score_raw = 0.8
    
    # Fuse scores
    fused = yolo_score
    if clip_enabled:
        fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw
    
    return fused  # Returns: 0.74

# New scoring with OCR integration:
def scoring_example_after(text_confidence=0.6, ocr_weight=0.25):
    """Enhanced scoring with OCR confidence"""
    yolo_score = 0.7
    clip_enabled = True
    clip_weight = 0.3
    clip_score_raw = 0.8
    
    # Fuse scores (YOLO + CLIP)
    fused = yolo_score
    if clip_enabled:
        fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw
    
    # Add OCR if available and confident enough
    if text_confidence > 0.0:
        fused = (1.0 - ocr_weight) * fused + ocr_weight * text_confidence
    
    return fused  # Returns: 0.7 (with text_conf=0.6)

# State machine threshold adjustment
def threshold_adjustment_example(text_confidence=0.6, base_enter_th=0.55):
    """Adjust enter threshold based on OCR confidence"""
    adjusted_enter_th = base_enter_th
    
    # If text is detected with high confidence, lower the threshold
    if text_confidence >= 0.7:
        adjusted_enter_th -= 0.10
    elif text_confidence >= 0.5:
        adjusted_enter_th -= 0.05
    
    return adjusted_enter_th  # Returns: 0.45 if text_conf=0.6

if __name__ == '__main__':
    print("Scoring Example (before):  {:.3f}".format(scoring_example_before()))
    print("Scoring Example (after):   {:.3f}".format(scoring_example_after(text_confidence=0.6)))
    print()
    print("Threshold adjustment: {:.3f} → {:.3f}".format(
        0.55, 
        threshold_adjustment_example(text_confidence=0.6)
    ))
