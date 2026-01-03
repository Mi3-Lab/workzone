"""
Multi-Cue Gate - Enforces AND logic on multiple cue types
Phase 1.1: Multi-Cue AND + Temporal Persistence
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path

from ..detection import FrameCues
from ..temporal import PersistenceState


@dataclass
class MultiCueDecision:
    """Decision output from multi-cue gate."""
    passed: bool  # Whether multi-cue gate passed
    num_sustained_cues: int  # Number of cues meeting persistence threshold
    sustained_cues: List[str]  # List of sustained cue groups
    confidence: float  # Confidence score (0.0-1.0)
    reason: str  # Human-readable reason


class MultiCueGate:
    """
    Enforces AND logic: work zone requires ≥2 independent cue types sustained.
    
    This is the core filter that eliminates false positives from single-cue
    detections (e.g., random cones in parking lot, orange truck alone).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize multi-cue gate.
        
        Args:
            config_path: Path to multi_cue_config.yaml
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
    
    def evaluate(
        self,
        frame_cues: FrameCues,
        persistence_states: Dict[str, PersistenceState]
    ) -> MultiCueDecision:
        """
        Evaluate whether multi-cue criteria are met.
        
        Args:
            frame_cues: Current frame cue detections
            persistence_states: Persistence state per cue group
            
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
        
        return MultiCueDecision(
            passed=passed,
            num_sustained_cues=num_sustained,
            sustained_cues=sustained_cues,
            confidence=min(confidence, 1.0),
            reason=reason
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
