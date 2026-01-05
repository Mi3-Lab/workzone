"""
OCR module for text extraction from work zone signs.

This module provides tools to extract and classify text from traffic signs,
message boards, and arrow boards detected in work zone videos.
"""

from .text_detector import SignTextDetector
from .text_classifier import TextClassifier

__all__ = ['SignTextDetector', 'TextClassifier']
