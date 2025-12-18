"""
WorkZone: AI-powered construction zone detection and monitoring system.

This package provides comprehensive tools for detecting, tracking, and analyzing
construction zones using YOLO object detection and vision-language models.
Part of the ESV (Embodied Scene Understanding for Vehicles) competition.
"""

__version__ = "1.0.0"
__author__ = "WMaia9"
__description__ = "Professional AI system for construction zone detection and analysis"

from src.workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)

__all__ = ["logger"]
