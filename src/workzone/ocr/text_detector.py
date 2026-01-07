"""
Text detection module using OCR backends (EasyOCR preferred, PaddleOCR optional).

Extracts text from cropped sign images with preprocessing and confidence scoring.
"""

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
            try:
                import paddle  # Lazy import
                from paddleocr import PaddleOCR  # Lazy import
            except ImportError as e:
                raise RuntimeError(f"PaddleOCR not installed: {e}")

            if use_gpu and hasattr(paddle, "is_compiled_with_cuda") and paddle.is_compiled_with_cuda():
                paddle.set_device("gpu")
                logger.info("Paddle set_device('gpu')")
            else:
                paddle.set_device("cpu")
                if use_gpu:
                    logger.warning("Paddle GPU not available, using CPU for OCR")
            ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            # paddle.get_device may not exist in some versions
            dev = getattr(paddle, "get_device", lambda: "cpu")()
            logger.info(f"PaddleOCR initialized (device={dev})")
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
                    # Lower threshold for better recall
                    if score >= 0.3:  # 30% confidence
                        text_clean = str(text).strip()
                        if len(text_clean) > 0:  # Skip empty strings
                            texts.append(text_clean)
                            confidences.append(float(score))
                if not texts:
                    return "", 0.0
                full_text = " ".join(texts)
                avg_conf = float(np.mean(confidences))
                conf_status = "low" if avg_conf < 0.5 else "ok"
                logger.info(f"OCR(Easy): \"{full_text}\" (conf: {avg_conf:.2f} [{conf_status}])")
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
                        if score >= 0.3:  # Lower threshold for better recall
                            text_clean = str(text).strip()
                            if len(text_clean) > 0:  # Skip empty strings
                                texts.append(text_clean)
                                confidences.append(float(score))
            
            if not texts:
                logger.debug("No text met confidence threshold")
                return "", 0.0
            
            # Join multi-line text
            full_text = " ".join(texts)
            avg_conf = float(np.mean(confidences))
            conf_status = "low" if avg_conf < 0.5 else "ok"
            
            logger.info(f"OCR(Paddle): \"{full_text}\" (conf: {avg_conf:.2f} [{conf_status}])")
            return full_text, avg_conf
            
        except Exception as e:
            import traceback
            logger.error(f"OCR extraction failed: {e}")
            logger.debug(traceback.format_exc())
            return "", 0.0
    
    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Enhance image for OCR.
        
        Applies multiple enhancement strategies:
        - Upscale small crops for better OCR
        - CLAHE for contrast enhancement
        - Sharpening filter
        - Denoising for very noisy images
        
        Args:
            crop: Input image (BGR or grayscale)
        
        Returns:
            Enhanced BGR image
        """
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        
        # Upscale small images (improves OCR significantly)
        h, w = crop.shape[:2]
        if h < 64 or w < 64:
            scale = max(2, 64 // min(h, w))
            crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        
        # Denoise if very noisy
        crop = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)
        
        # CLAHE on luminance channel
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen to enhance text edges
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
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
