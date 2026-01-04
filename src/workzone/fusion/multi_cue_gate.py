"""
Multi-Cue Gate - Enforces AND logic on multiple cue types
Phase 1.1: Multi-Cue AND + Temporal Persistence
Phase 1.3: Motion plausibility validation (optional)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path
import numpy as np

from ..detection import FrameCues
from ..temporal import PersistenceState
from ..types import MultiCueDecision


class MultiCueGate:
    """
    Enforces AND logic: work zone requires ≥1 independent cue type sustained.
    
    Optional Phase 1.3: Motion validation for additional false positive filtering.
    """
    
    def __init__(self, config_path: Optional[str] = None, enable_motion: bool = False):
        """
        Initialize multi-cue gate.
        
        Args:
            config_path: Path to multi_cue_config.yaml
            enable_motion: Enable Phase 1.3 motion validation
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "multi_cue_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get multi-cue parameters
        gate_config = self.config['multi_cue_gate']
        self.min_cues = gate_config['min_sustained_cues']
        self.confidence_per_cue = gate_config['confidence_per_cue']
        self.base_confidence = gate_config['base_confidence']
        
        # Get cue weights (for confidence scoring)
        self.cue_weights = self.config['temporal']['cue_weights']
        
        # Phase 1.3 motion validation (optional)
        self.enable_motion = enable_motion
        self.motion_gate = None
        if enable_motion:
            try:
                from ..motion import MotionCueGate
                motion_config = Path(__file__).parent.parent.parent.parent / "configs" / "motion_cue_config.yaml"
                self.motion_gate = MotionCueGate()
            except ImportError:
                print("⚠ Motion module not available, disabling Phase 1.3")
                self.enable_motion = False
    
    def evaluate(
        self,
        frame_cues: FrameCues,
        persistence_states: Dict[str, PersistenceState],
        frame: Optional[np.ndarray] = None,
        yolo_results = None
    ) -> MultiCueDecision:
        """
        Evaluate whether multi-cue criteria are met.
        
        Args:
            frame_cues: Current frame cue detections
            persistence_states: Persistence state per cue group
            frame: Optional frame for Phase 1.3 motion validation
            yolo_results: Optional YOLO results for motion validation
            
        Returns:
            MultiCueDecision with pass/fail and metadata
        """
        # Get sustained cues
        sustained_cues = [
            group for group, state in persistence_states.items()
            if state.is_sustained
        ]
        
        num_sustained = len(sustained_cues)
        
        # Check multi-cue requirement
        passed = num_sustained >= self.min_cues
        
        # Compute confidence score
        if num_sustained == 0:
            confidence = 0.0
            reason = "No sustained cues detected"
        elif num_sustained < self.min_cues:
            confidence = 0.3
            reason = f"Only {num_sustained} cue(s) sustained ({', '.join(sustained_cues)}), need ≥{self.min_cues}"
        else:
            # Base confidence + boost per additional cue
            confidence = self.base_confidence + (num_sustained - self.min_cues) * self.confidence_per_cue
            confidence = min(confidence, 1.0)
            reason = f"{num_sustained} cues sustained: {', '.join(sustained_cues)}"
        
        # Adjust confidence by cue weights and persistence scores
        if num_sustained > 0:
            weighted_score = sum(
                self.cue_weights[cue] * persistence_states[cue].persistence_score
                for cue in sustained_cues
            ) / num_sustained
            
            # Blend with count-based confidence
            confidence = 0.7 * confidence + 0.3 * min(weighted_score, 1.0)
        
        # Phase 1.3: Optional motion validation
        motion_validated = True
        motion_plausibility = 1.0
        
        if self.enable_motion and self.motion_gate is not None and frame is not None and yolo_results is not None:
            try:
                motion_result = self.motion_gate.evaluate(frame, yolo_results)
                motion_validated = motion_result.is_plausible
                motion_plausibility = motion_result.confidence
                
                # Apply motion penalty to confidence if validation fails
                if not motion_validated:
                    confidence *= 0.7  # Reduce confidence by 30%
                    reason += f" [Motion: {motion_result.motion_type}]"
            except Exception as e:
                print(f"⚠ Motion validation error: {e}")
        
        return MultiCueDecision(
            passed=passed and motion_validated,
            num_sustained_cues=num_sustained,
            sustained_cues=sustained_cues,
            confidence=min(confidence, 1.0),
            reason=reason,
            motion_validated=motion_validated,
            motion_plausibility=motion_plausibility
        )
    
    def get_required_cues(self) -> int:
        """Get minimum number of cues required."""
        return self.min_cues


# ===================================
# UTILITY FUNCTIONS
# ===================================

def print_decision(decision: MultiCueDecision):
    """Print multi-cue decision for debugging."""
    status = "✓ PASSED" if decision.passed else "✗ FAILED"
    print(f"\n=== MULTI-CUE GATE {status} ===")
    print(f"  Sustained cues: {decision.num_sustained_cues}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Reason: {decision.reason}")
    if decision.sustained_cues:
        print(f"  Cues: {', '.join(decision.sustained_cues)}")


if __name__ == "__main__":
    # Test multi-cue gate
    gate = MultiCueGate()
    print(f"\n✓ Multi-Cue Gate initialized")
    print(f"  - Min cues required: {gate.min_cues}")
    print(f"  - High confidence threshold: {gate.high_conf_cues} cues")
