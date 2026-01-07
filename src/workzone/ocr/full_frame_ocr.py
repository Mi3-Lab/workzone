"""
Full-frame OCR detection - independent of YOLO.
Detects text anywhere in the frame, not just in YOLO-detected regions.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def detect_text_regions_fullframe(
    frame: np.ndarray,
    detector,
    min_confidence: float = 0.3,
    min_text_length: int = 2,
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """
    Detect all text regions in full frame using OCR text detection.
    
    Args:
        frame: Input frame
        detector: OCR detector (EasyOCR or PaddleOCR)
        min_confidence: Minimum confidence threshold
        min_text_length: Minimum text length to consider
    
    Returns:
        List of (text, confidence, bbox) tuples
        bbox is (x1, y1, x2, y2)
    """
    if frame is None or frame.size == 0:
        return []
    
    try:
        # Use EasyOCR's built-in text detection which returns bounding boxes
        backend = getattr(detector, "backend", "paddle")
        
        if backend == "easyocr":
            reader = detector.easy
            # EasyOCR returns: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence), ...]
            results = reader.readtext(frame, detail=1, paragraph=False)
            
            detections = []
            for bbox_points, text, conf in results:
                if conf >= min_confidence and len(text.strip()) >= min_text_length:
                    # Convert polygon to axis-aligned bbox
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                    detections.append((text.strip(), float(conf), (x1, y1, x2, y2)))
            
            return detections
        
        else:
            # PaddleOCR path - similar logic
            result = detector.ocr.ocr(frame)
            
            detections = []
            if result and len(result) > 0:
                for page_result in result:
                    if isinstance(page_result, dict):
                        rec_texts = page_result.get('rec_texts', [])
                        rec_scores = page_result.get('rec_scores', [])
                        boxes = page_result.get('dt_boxes', [])
                        
                        for text, score, box in zip(rec_texts, rec_scores, boxes):
                            if score >= min_confidence and len(text.strip()) >= min_text_length:
                                # Convert polygon to bbox
                                xs = [p[0] for p in box]
                                ys = [p[1] for p in box]
                                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                                detections.append((text.strip(), float(score), (x1, y1, x2, y2)))
            
            return detections
    
    except Exception as e:
        logger.error(f"Full-frame OCR detection error: {e}")
        return []


def extract_best_text_fullframe(
    frame: np.ndarray,
    detector,
    classifier,
    corrector=None,
    aggregator=None,
    frame_idx: int = 0,
    min_confidence: float = 0.3,
    threshold: float = 0.25,
) -> Tuple[str, float, str, List[Dict]]:
    """
    Extract best text from full frame (YOLO-independent).
    
    Args:
        frame: Input frame
        detector: OCR detector
        classifier: Text classifier
        corrector: Optional spell corrector
        aggregator: Optional temporal aggregator
        frame_idx: Current frame index
        min_confidence: Minimum confidence threshold
        threshold: Alias for min_confidence (for compatibility)
    
    Returns:
        (best_text, confidence, category, all_detections)
        all_detections is list of dicts with text/conf/bbox/category
    """
    # Use threshold parameter if provided, otherwise use min_confidence
    effective_threshold = threshold if threshold != 0.25 or min_confidence == 0.3 else min_confidence
    
    # Detect all text regions in frame
    detections = detect_text_regions_fullframe(
        frame, detector, min_confidence=effective_threshold
    )
    
    if not detections:
        logger.debug(f"Frame {frame_idx}: No text detected (threshold={effective_threshold})")
        return "", 0.0, "NONE", []
    
    logger.debug(f"Frame {frame_idx}: Found {len(detections)} text regions in full frame")
    
    # Process each detection
    all_results = []
    for text, conf, bbox in detections:
        logger.debug(f"OCR detected: '{text}' (conf={conf:.2f})")
        # Apply spell correction
        corrected_text = text
        correction_factor = 1.0
        if corrector:
            corrected_text, correction_factor = corrector.correct_text(text)
        
        # Classify
        category, class_conf = classifier.classify(corrected_text)
        
        # Final confidence
        final_conf = conf * class_conf * correction_factor
        
        all_results.append({
            'text': corrected_text,
            'confidence': final_conf,
            'category': category,
            'bbox': bbox,
            'raw_text': text,
        })
        
        logger.debug(f"  Text: '{corrected_text}' cat={category} conf={final_conf:.2f} bbox={bbox}")
    
    # Temporal aggregation on best result
    if all_results:
        # PRIORITY: Sort speed signs to top, then by confidence
        def sort_priority(x):
            # Speed signs get highest priority (boost confidence by 0.5)
            if x['category'] == 'SPEED':
                return x['confidence'] + 0.5
            # Workzone/lane/caution signs get medium priority (boost by 0.2)
            elif x['category'] in ['WORKZONE', 'LANE', 'CAUTION']:
                return x['confidence'] + 0.2
            # Everything else normal
            return x['confidence']
        
        all_results.sort(key=sort_priority, reverse=True)
        best = all_results[0]
        
        if aggregator:
            aggregator.add_detection(best['text'], best['confidence'], frame_idx)
            agg_text, agg_conf = aggregator.get_best_text(frame_idx)
            if agg_conf > best['confidence'] * 0.8:
                # Re-classify aggregated text
                agg_cat, agg_class_conf = classifier.classify(agg_text)
                return agg_text, agg_conf, agg_cat, all_results
        
        return best['text'], best['confidence'], best['category'], all_results
    
    return "", 0.0, "NONE", []
