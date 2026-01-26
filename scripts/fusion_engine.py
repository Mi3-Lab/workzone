# Phase 2: SOTA Fusion Engine
import math

def clamp01(x):
    return max(0.0, min(1.0, x))

def fuse_states(yolo_score, vlm_result, current_fused_score):
    """
    Implements the SOTA fusion logic between YOLO and VLM.
    
    Args:
        yolo_score (float): The raw semantic score from YOLO detections.
        vlm_result (dict): The JSON output from the VLM.
        current_fused_score (float): The current smoothed score of the system.
        
    Returns:
        float: The new, fused score.
    """
    #
    # Default to YOLO score if VLM is not providing data
    if not vlm_result or "derived_state" not in vlm_result:
        return yolo_score

    vlm_state = vlm_result.get("derived_state", {}).get("state", "OUT")
    vlm_confidence = vlm_result.get("derived_state", {}).get("confidence", 0.0)
    
    # --- SOTA Fusion Rules ---
    
    # Rule 1: VLM provides a strong VETO.
    # If VLM is highly confident it's OUT, significantly reduce the score.
    if vlm_state == "OUT" and vlm_confidence > 0.90:
        # Heavily penalize the score, but don't drop to zero instantly.
        # This acts as a strong "brake" on false positives.
        return current_fused_score * 0.1
        
    # Rule 2: Strong Agreement (YOLO score is high AND VLM agrees)
    # This boosts confidence towards 1.0.
    if yolo_score > 0.6 and vlm_state in ["APPROACHING", "INSIDE"] and vlm_confidence > 0.7:
        # Combine scores and add a bonus for agreement
        agreement_score = (yolo_score + vlm_confidence) / 2.0
        return clamp01(agreement_score + 0.15)
        
    # Rule 3: VLM detects what YOLO might be missing (Safety Fallback)
    # If VLM sees a workzone but YOLO score is low, trust the VLM.
    if yolo_score < 0.4 and vlm_state in ["APPROACHING", "INSIDE"] and vlm_confidence > 0.8:
        # The new score is primarily driven by the VLM's confidence.
        return vlm_confidence * 0.8
        
    # Default Case: No strong signals, just blend the scores.
    # Give more weight to the instantaneous YOLO score to be reactive.
    # This is a simple blend; in the future, EMA of each score could be used.
    fused_score = (yolo_score * 0.8) + (vlm_confidence * 0.2)
    
    return clamp01(fused_score)
