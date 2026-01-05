"""
Text detection module using PaddleOCR.

Extracts text from cropped sign images with preprocessing and confidence scoring.
"""

from paddleocr import PaddleOCR
import paddle
try:
    import easyocr
    EASY_AVAILABLE = True
except ImportError:
    EASY_AVAILABLE = False
import torch
import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SignTextDetector:
    """
    PaddleOCR-based text detector for work zone signs.
    
    Optimized for message boards and arrow boards with text content.
    Includes preprocessing to enhance OCR accuracy.
    """
    
    def __init__(self, use_gpu: bool = True, lang: str = 'en', verbose: bool = False, prefer_easyocr: bool = True):
        """
        Initialize PaddleOCR detector.
        
        Args:
            use_gpu: Whether to use GPU acceleration (CUDA) - only for PaddlePaddle backend
            lang: Language code ('en' for English)
            verbose: Enable PaddleOCR logging (not used in PaddleOCR 3.x)
        """
        self.backend = None
        self.ocr = None
        self.easy = None

        last_error = None

        def init_easy():
            gpu_flag = use_gpu and torch.cuda.is_available()
            reader = easyocr.Reader([lang], gpu=gpu_flag)
            logger.info(f"EasyOCR initialized (gpu={gpu_flag})")
            return reader

        def init_paddle():
            if use_gpu and paddle.is_compiled_with_cuda():
                paddle.set_device("gpu")
                logger.info("Paddle set_device('gpu')")
            else:
                paddle.set_device("cpu")
                if use_gpu:
                    logger.warning("Paddle GPU not available, using CPU for OCR")
            ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            logger.info(f"PaddleOCR initialized (device={paddle.get_device()})")
            return ocr

        # Prefer EasyOCR when available (avoids Paddle import issues)
        if prefer_easyocr and EASY_AVAILABLE:
            try:
                self.easy = init_easy()
                self.backend = "easyocr"
                return
            except Exception as e:
                last_error = e
                logger.warning(f"EasyOCR init failed, will try Paddle: {e}")

        # Try PaddleOCR next
        try:
            self.ocr = init_paddle()
            self.backend = "paddle"
            return
        except Exception as e:
            last_error = e
            logger.error(f"PaddleOCR init failed: {e}")

        # Fallback to EasyOCR if Paddle failed and EasyOCR is present
        if EASY_AVAILABLE:
            try:
                self.easy = init_easy()
                self.backend = "easyocr"
                return
            except Exception as e:
                last_error = e
                logger.error(f"EasyOCR fallback failed: {e}")

        # If nothing worked, surface the last error
        raise RuntimeError(f"Failed to initialize OCR backend: {last_error}")
    
    def extract_text(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from a sign crop.
        
        Args:
            crop: Image crop (BGR or grayscale numpy array)
        
        Returns:
            (text, confidence): Extracted text and average confidence [0-1]
                                Returns ("", 0.0) if no text detected
        """
        if crop is None or crop.size == 0:
            return "", 0.0
        
        # Check minimum size
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            logger.warning(f"Crop too small for OCR: {crop.shape}")
            return "", 0.0
        
        # Enhance contrast for better OCR
        enhanced = self._preprocess(crop)
        
        try:
            # PaddleOCR 3.x returns a list of dicts with structured output
            if getattr(self, "backend", "paddle") == "easyocr" and getattr(self, "easy", None) is not None:
                # EasyOCR path
                result = self.easy.readtext(enhanced, detail=1, paragraph=False)
                if not result:
                    return "", 0.0
                texts = []
                confidences = []
                for _, text, score in result:
                    if score >= 0.50:
                        texts.append(str(text))
                        confidences.append(float(score))
                if not texts:
                    return "", 0.0
                full_text = " ".join(texts)
                avg_conf = float(np.mean(confidences))
                logger.info(f"OCR(Easy): \"{full_text}\" (conf: {avg_conf:.2f})")
                return full_text, avg_conf

            # Paddle path
            result = self.ocr.ocr(enhanced)
            
            # Handle empty results
            if not result or len(result) == 0:
                logger.debug("OCR returned empty result")
                return "", 0.0
            
            # Extract text from the new structure
            texts = []
            confidences = []
            
            # Result is a list of dicts, each containing OCR results
            for i, page_result in enumerate(result):
                if isinstance(page_result, dict):
                    # New format has rec_texts and rec_scores
                    rec_texts = page_result.get('rec_texts', [])
                    rec_scores = page_result.get('rec_scores', [])
                    
                    logger.debug(f"Page {i}: {len(rec_texts)} texts found")
                    
                    for text, score in zip(rec_texts, rec_scores):
                        if score >= 0.50:  # Confidence filter
                            texts.append(str(text))
                            confidences.append(float(score))
            
            if not texts:
                logger.debug("No text met confidence threshold")
                return "", 0.0
            
            # Join multi-line text
            full_text = " ".join(texts)
            avg_conf = float(np.mean(confidences))
            
            logger.info(f"OCR: \"{full_text}\" (conf: {avg_conf:.2f})")
            return full_text, avg_conf
            
        except Exception as e:
            import traceback
            logger.error(f"OCR extraction failed: {e}")
            logger.debug(traceback.format_exc())
            return "", 0.0
    
    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Enhance image for OCR.
        
        Applies:
        - Grayscale conversion
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Slight denoising
        
        Args:
            crop: Input image (BGR or grayscale)
        
        Returns:
            Enhanced grayscale or BGR image (PaddleOCR 3.x prefers BGR)
        """
        # PaddleOCR 3.x actually works better with BGR, not grayscale
        # Just apply CLAHE on the luminance channel
        if len(crop.shape) == 2:
            # Grayscale - convert to BGR
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        
        # Enhance contrast using LAB color space
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def extract_text_batch(self, crops: list) -> list:
        """
        Extract text from multiple crops (batch processing).
        
        Args:
            crops: List of image crops
        
        Returns:
            List of (text, confidence) tuples
        """
        results = []
        for crop in crops:
            text, conf = self.extract_text(crop)
            results.append((text, conf))
        return results
