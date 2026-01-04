"""
Shared types and dataclasses for WorkZone detection pipeline.

This module provides common types to avoid circular import dependencies.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MultiCueDecision:
    """Decision output from multi-cue gate."""
    passed: bool  # Whether multi-cue gate passed
    num_sustained_cues: int  # Number of cues meeting persistence threshold
    sustained_cues: List[str]  # List of sustained cue groups
    confidence: float  # Confidence score (0.0-1.0)
    reason: str  # Human-readable reason
    motion_validated: bool = True  # Whether motion validation passed (default True if not used)
    motion_plausibility: float = 1.0  # Motion plausibility score if validated
