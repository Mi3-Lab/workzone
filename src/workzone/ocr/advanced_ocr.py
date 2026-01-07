"""
Advanced OCR system with SOTA techniques for workzone text extraction.

Features:
- Multi-frame temporal aggregation (voting across frames)
- Spell correction with workzone vocabulary
- Enhanced preprocessing (super-resolution, perspective correction)
- Confidence-aware fusion with YOLO detections
- Context-aware text validation
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from difflib import get_close_matches
import logging

logger = logging.getLogger(__name__)


# Comprehensive workzone vocabulary for spell correction
WORKZONE_VOCABULARY = {
    # Common workzone messages
    "ROAD", "WORK", "AHEAD", "ZONE", "WORKERS", "WORKING",
    "CAUTION", "WARNING", "DANGER", "SLOW", "STOP", "GO",
    "DETOUR", "CLOSED", "CONSTRUCTION", "MAINTENANCE",
    
    # Speed and traffic
    "SPEED", "LIMIT", "MPH", "KMH", "REDUCED", "MINIMUM", "MAXIMUM",
    "REDUCED", "ENFORCED",
    
    # Directional
    "LEFT", "RIGHT", "STRAIGHT", "AHEAD", "EXIT", "ENTER",
    "LANE", "LANES", "MERGE", "SHIFT", "KEEP", "USE", "ONLY",
    "ENDS", "BEGINS", "CLOSED", "OPEN",
    
    # Time/distance
    "MILES", "FEET", "NEXT", "MILES", "KM", "METERS",
    "HOURS", "DAYS", "TEMPORARY", "PERMANENT",
    
    # Equipment/personnel
    "FLAGGER", "FLAGGING", "PILOT", "VEHICLE", "TRUCK", "EQUIPMENT",
    "CREW", "PERSONNEL",
    
    # Common numbers as words
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "TEN", "TWENTY", "THIRTY",
    "FORTY", "FIFTY", "SIXTY", "SEVENTY",
    
    # Common misspellings observed
    "WRORK", "AHED", "AHAED", "AMEAD", "CAON", "SPPED",
}

# Regex patterns for validation
SPEED_PATTERN = re.compile(r'\b\d{1,3}\s*(?:MPH|KMH|LIMIT)?\b', re.IGNORECASE)
DISTANCE_PATTERN = re.compile(r'\b\d+\s*(?:MILES?|FEET|KM|METERS?)\b', re.IGNORECASE)
DIRECTION_KEYWORDS = ["LEFT", "RIGHT", "AHEAD", "EXIT", "ENTER", "MERGE", "LANE", "SHIFT"]
WORKZONE_KEYWORDS = ["WORK", "ROAD", "CONSTRUCTION", "WORKER", "CAUTION", "ZONE"]


class TemporalOCRAggregator:
    """
    Aggregates OCR results across multiple frames for improved accuracy.
    Uses voting and confidence weighting to select best text.
    """
    
    def __init__(self, window_size: int = 30, similarity_threshold: float = 0.7):
        """
        Args:
            window_size: Number of frames to keep in history
            similarity_threshold: Minimum similarity to consider texts as same
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.history: List[Tuple[str, float, int]] = []  # (text, confidence, frame_idx)
    
    def add_detection(self, text: str, confidence: float, frame_idx: int):
        """Add new OCR detection to history."""
        if text and len(text.strip()) > 0:
            self.history.append((text.strip().upper(), confidence, frame_idx))
            # Keep only recent frames
            if len(self.history) > self.window_size:
                self.history = self.history[-self.window_size:]
    
    def get_best_text(self, current_frame: int, recency_weight: float = 0.3) -> Tuple[str, float]:
        """
        Get most confident text considering temporal voting and recency.
        
        Args:
            current_frame: Current frame index
            recency_weight: Weight for recency (0-1, higher = prefer recent)
        
        Returns:
            (best_text, aggregated_confidence)
        """
        if not self.history:
            return "", 0.0
        
        # Group similar texts
        text_groups = defaultdict(list)
        for text, conf, frame_idx in self.history:
            # Find similar group or create new
            matched = False
            for key in text_groups.keys():
                if self._texts_similar(text, key):
                    text_groups[key].append((text, conf, frame_idx))
                    matched = True
                    break
            if not matched:
                text_groups[text].append((text, conf, frame_idx))
        
        # Score each group
        best_text = ""
        best_score = 0.0
        
        for representative, detections in text_groups.items():
            # Compute weighted score
            total_conf = 0.0
            total_weight = 0.0
            
            for text, conf, frame_idx in detections:
                # Recency weight (exponential decay)
                frame_diff = current_frame - frame_idx
                recency = np.exp(-recency_weight * frame_diff / self.window_size)
                
                weight = conf * recency
                total_conf += weight
                total_weight += recency
            
            # Normalize by number of detections and weight
            avg_score = total_conf / max(1, total_weight) if total_weight > 0 else 0.0
            # Boost by frequency
            frequency_boost = 1.0 + 0.1 * len(detections)
            final_score = avg_score * frequency_boost
            
            if final_score > best_score:
                best_score = final_score
                # Use most common variant in group
                best_text = Counter([t for t, _, _ in detections]).most_common(1)[0][0]
        
        return best_text, min(1.0, best_score)
    
    def _texts_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar enough to be grouped."""
        # Exact match
        if text1 == text2:
            return True
        
        # Edit distance based similarity
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, text1, text2).ratio()
        return ratio >= self.similarity_threshold
    
    def clear(self):
        """Clear history (use when scene changes)."""
        self.history.clear()


class WorkzoneSpellCorrector:
    """
    Spell correction specialized for workzone text using vocabulary and context.
    """
    
    def __init__(self, vocabulary: set = None):
        self.vocabulary = vocabulary or WORKZONE_VOCABULARY
        # Add lowercase versions
        self.vocabulary = {w.upper() for w in self.vocabulary} | {w.lower() for w in self.vocabulary}
    
    def correct_text(self, text: str, context_hint: str = None) -> Tuple[str, float]:
        """
        Correct spelling errors in OCR text.
        
        Args:
            text: Raw OCR text
            context_hint: Optional context (e.g., "WORKZONE", "SPEED_LIMIT")
        
        Returns:
            (corrected_text, confidence_factor) where factor is 0-1 penalty for corrections
        """
        if not text or len(text.strip()) == 0:
            return "", 0.0
        
        words = text.upper().split()
        corrected_words = []
        total_penalty = 0.0
        
        for word in words:
            # Keep numbers as-is
            if word.isdigit() or any(c.isdigit() for c in word):
                corrected_words.append(word)
                continue
            
            # If word in vocabulary, keep it
            if word in self.vocabulary or len(word) <= 2:
                corrected_words.append(word)
                continue
            
            # Try to find close match
            matches = get_close_matches(word, self.vocabulary, n=1, cutoff=0.75)
            if matches:
                corrected_words.append(matches[0])
                # Penalty based on edit distance
                penalty = 1.0 - SequenceMatcher(None, word, matches[0]).ratio()
                total_penalty += penalty * 0.2  # Reduce penalty impact
                logger.debug(f"Corrected '{word}' → '{matches[0]}' (penalty: {penalty:.2f})")
            else:
                # Keep original if no good match
                corrected_words.append(word)
                total_penalty += 0.1  # Small penalty for unknown word
        
        corrected = " ".join(corrected_words)
        confidence_factor = max(0.0, 1.0 - total_penalty / max(1, len(words)))
        
        return corrected, confidence_factor


class EnhancedPreprocessor:
    """
    Advanced image preprocessing for better OCR accuracy.
    """
    
    @staticmethod
    def enhance_crop(crop: np.ndarray, target_height: int = 128) -> np.ndarray:
        """
        Apply multiple enhancement techniques to improve OCR.
        
        Args:
            crop: Input image
            target_height: Target height for upscaling (if small)
        
        Returns:
            Enhanced image
        """
        if crop is None or crop.size == 0:
            return crop
        
        h, w = crop.shape[:2]
        
        # 1. Upscale if too small (super-resolution via bicubic)
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            crop = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
            h, w = crop.shape[:2]
        
        # 2. Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 4. Adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 5. Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 6. Adaptive thresholding for high contrast
        # (Optional: can help with certain sign types)
        # thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY, 11, 2)
        
        return sharpened
    
    @staticmethod
    def detect_and_correct_perspective(crop: np.ndarray) -> np.ndarray:
        """
        Detect and correct perspective distortion in sign images.
        (Simplified version - can be expanded with more sophisticated methods)
        """
        # This is a placeholder for more advanced perspective correction
        # Could use:
        # - Hough lines to detect sign boundaries
        # - Contour detection + perspective transform
        # - Deep learning based perspective estimation
        return crop


from difflib import SequenceMatcher


def advanced_ocr_pipeline(
    crop: np.ndarray,
    detector,
    classifier,
    aggregator: Optional[TemporalOCRAggregator] = None,
    corrector: Optional[WorkzoneSpellCorrector] = None,
    frame_idx: int = 0,
    yolo_confidence: float = 1.0,
) -> Tuple[str, float, str]:
    """
    Advanced OCR pipeline with all enhancements.
    
    Args:
        crop: Image crop to process
        detector: OCR detector (SignTextDetector)
        classifier: Text classifier
        aggregator: Optional temporal aggregator
        corrector: Optional spell corrector
        frame_idx: Current frame index
        yolo_confidence: YOLO detection confidence (boosts OCR confidence)
    
    Returns:
        (text, confidence, category)
    """
    if crop is None or crop.size == 0:
        return "", 0.0, "NONE"
    
    # 1. Enhanced preprocessing
    preprocessor = EnhancedPreprocessor()
    enhanced = preprocessor.enhance_crop(crop, target_height=128)
    
    # 2. Extract text with base OCR
    raw_text, ocr_conf = detector.extract_text(enhanced)
    
    if not raw_text or ocr_conf < 0.25:
        return "", 0.0, "NONE"
    
    # 3. Spell correction
    corrected_text = raw_text
    correction_factor = 1.0
    if corrector:
        corrected_text, correction_factor = corrector.correct_text(raw_text)
        if corrected_text != raw_text:
            logger.info(f"Spell correction: '{raw_text}' → '{corrected_text}'")
    
    # 4. Temporal aggregation
    if aggregator:
        aggregator.add_detection(corrected_text, ocr_conf * correction_factor, frame_idx)
        # Use aggregated result if confidence is higher
        agg_text, agg_conf = aggregator.get_best_text(frame_idx)
        if agg_conf > ocr_conf * correction_factor * 0.8:  # Use if significantly better
            corrected_text = agg_text
            ocr_conf = agg_conf
            logger.debug(f"Using temporal aggregation: '{corrected_text}' (conf: {agg_conf:.2f})")
    
    # 5. Classify text
    text_category, class_conf = classifier.classify(corrected_text)
    
    # 6. Context-aware validation and confidence boosting
    final_confidence = ocr_conf * class_conf * correction_factor * yolo_confidence
    
    # Boost confidence for validated patterns
    norm = corrected_text.upper()
    
    # Speed limit detection
    if SPEED_PATTERN.search(norm):
        if text_category in ["UNCLEAR", "NONE"]:
            text_category = "SPEED_LIMIT"
        final_confidence = min(1.0, final_confidence * 1.2)
        logger.info(f"Speed limit detected: '{corrected_text}'")
    
    # Workzone keywords
    if any(kw in norm for kw in WORKZONE_KEYWORDS):
        if text_category in ["UNCLEAR", "NONE"]:
            text_category = "WORKZONE"
        final_confidence = min(1.0, final_confidence * 1.3)
        logger.info(f"Workzone keyword detected: '{corrected_text}'")
    
    # Directional signs
    if any(kw in norm for kw in DIRECTION_KEYWORDS):
        if text_category in ["UNCLEAR", "NONE"]:
            text_category = "DIRECTION"
        final_confidence = min(1.0, final_confidence * 1.1)
    
    # Distance markers
    if DISTANCE_PATTERN.search(norm):
        final_confidence = min(1.0, final_confidence * 1.15)
    
    return corrected_text, final_confidence, text_category
