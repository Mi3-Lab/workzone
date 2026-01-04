"""
State Machine - Manages work zone detection states
Phase 1.1: Multi-Cue AND + Temporal Persistence

States: OUT → APPROACHING → INSIDE → EXITING → OUT
"""

from typing import Optional
from enum import Enum
from dataclasses import dataclass
import yaml
from pathlib import Path

from ..detection import FrameCues
from ..temporal import PersistenceState
from ..types import MultiCueDecision


class WorkZoneState(Enum):
    """Work zone detection states."""
    OUT = "OUT"
    APPROACHING = "APPROACHING"
    INSIDE = "INSIDE"
    EXITING = "EXITING"


@dataclass
class StateTransition:
    """Record of a state transition."""
    frame_id: int
    timestamp: float
    from_state: WorkZoneState
    to_state: WorkZoneState
    reason: str
    num_sustained_cues: int
    confidence: float


class WorkZoneStateMachine:
    """
    Manages state transitions for work zone detection.
    
    State flow:
    - OUT: No work zone detected
    - APPROACHING: Initial detection, single cue or low persistence
    - INSIDE: Work zone confirmed (≥2 cues sustained)
    - EXITING: Leaving work zone (cues decreasing)
    
    Hysteresis: Harder to enter INSIDE than to stay INSIDE (prevents flicker)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize state machine.
        
        Args:
            config_path: Path to multi_cue_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "multi_cue_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get state machine parameters
        sm_config = self.config['state_machine']
        self.transitions_config = sm_config['transitions']
        self.initial_state = WorkZoneState[sm_config['initial_state']]
        
        # Initialize state
        self.current_state = self.initial_state
        self.frames_in_state = 0
        self.exit_cooldown_counter = 0
        
        # State history
        self.state_history = []
        self.transitions = []
    
    def update(
        self,
        frame_cues: FrameCues,
        persistence_states: dict,
        multi_cue_decision: MultiCueDecision
    ) -> WorkZoneState:
        """
        Update state machine with new frame data.
        
        Args:
            frame_cues: Current frame cue detections
            persistence_states: Persistence state per cue group
            multi_cue_decision: Multi-cue gate decision
            
        Returns:
            Current state after update
        """
        previous_state = self.current_state
        num_sustained = multi_cue_decision.num_sustained_cues
        
        # Compute average persistence of sustained cues
        if num_sustained > 0:
            avg_persistence = sum(
                state.persistence_score
                for cue, state in persistence_states.items()
                if state.is_sustained
            ) / num_sustained
        else:
            avg_persistence = 0.0
        
        # State transition logic
        new_state = self._determine_next_state(
            num_sustained,
            avg_persistence,
            multi_cue_decision.passed
        )
        
        # Handle state change
        if new_state != self.current_state:
            transition = StateTransition(
                frame_id=frame_cues.frame_id,
                timestamp=frame_cues.timestamp,
                from_state=self.current_state,
                to_state=new_state,
                reason=multi_cue_decision.reason,
                num_sustained_cues=num_sustained,
                confidence=multi_cue_decision.confidence
            )
            self.transitions.append(transition)
            self.current_state = new_state
            self.frames_in_state = 0
        else:
            self.frames_in_state += 1
        
        # Record state history
        self.state_history.append(self.current_state)
        
        return self.current_state
    
    def _determine_next_state(
        self,
        num_sustained: int,
        avg_persistence: float,
        multi_cue_passed: bool
    ) -> WorkZoneState:
        """
        Determine next state based on current state and cue data.
        
        Args:
            num_sustained: Number of sustained cues
            avg_persistence: Average persistence score
            multi_cue_passed: Whether multi-cue gate passed
            
        Returns:
            Next state
        """
        if self.current_state == WorkZoneState.OUT:
            # OUT → APPROACHING: Detection begins
            trans = self.transitions_config['out_to_approaching']
            if num_sustained >= trans['min_sustained_cues']:
                return WorkZoneState.APPROACHING
            return WorkZoneState.OUT
        
        elif self.current_state == WorkZoneState.APPROACHING:
            # APPROACHING → INSIDE: Multi-cue confidence high
            trans_inside = self.transitions_config['approaching_to_inside']
            if multi_cue_passed and num_sustained >= trans_inside['min_sustained_cues']:
                return WorkZoneState.INSIDE
            
            # APPROACHING → OUT: Lost detection
            trans_out = self.transitions_config['out_to_approaching']
            if num_sustained < trans_out['min_sustained_cues']:
                return WorkZoneState.OUT
            
            return WorkZoneState.APPROACHING
        
        elif self.current_state == WorkZoneState.INSIDE:
            # INSIDE → EXITING: Cue count drops (hysteresis)
            trans = self.transitions_config['inside_to_exiting']
            if num_sustained <= trans['max_sustained_cues']:
                return WorkZoneState.EXITING
            
            return WorkZoneState.INSIDE
        
        elif self.current_state == WorkZoneState.EXITING:
            # EXITING → INSIDE: Cues return
            trans_inside = self.transitions_config['approaching_to_inside']
            if multi_cue_passed and num_sustained >= trans_inside['min_sustained_cues']:
                self.exit_cooldown_counter = 0
                return WorkZoneState.INSIDE
            
            # EXITING → OUT: Cooldown complete
            trans_out = self.transitions_config['exiting_to_out']
            cooldown = trans_out['exit_cooldown_frames']
            
            if num_sustained == 0:
                self.exit_cooldown_counter += 1
                if self.exit_cooldown_counter >= cooldown:
                    self.exit_cooldown_counter = 0
                    return WorkZoneState.OUT
            else:
                self.exit_cooldown_counter = 0
            
            return WorkZoneState.EXITING
        
        return self.current_state
    
    def get_state_confidence(self) -> float:
        """Get confidence score for current state."""
        # Return confidence based on frames in state
        if self.frames_in_state < 5:
            return 0.6
        elif self.frames_in_state < 15:
            return 0.8
        else:
            return 0.95
    
    def reset(self):
        """Reset state machine to initial state."""
        self.current_state = self.initial_state
        self.frames_in_state = 0
        self.exit_cooldown_counter = 0
        self.state_history.clear()
        self.transitions.clear()
    
    def get_transition_summary(self) -> dict:
        """Get summary of all state transitions."""
        return {
            'total_transitions': len(self.transitions),
            'transitions': [
                {
                    'frame': t.frame_id,
                    'time': f"{t.timestamp:.2f}s",
                    'transition': f"{t.from_state.value} → {t.to_state.value}",
                    'reason': t.reason,
                    'confidence': f"{t.confidence:.2f}"
                }
                for t in self.transitions
            ]
        }



# ===================================
# UTILITY FUNCTIONS
# ===================================

def print_state(state_machine: WorkZoneStateMachine):
    """Print current state for debugging."""
    state = state_machine.current_state
    confidence = state_machine.get_state_confidence()
    frames = state_machine.frames_in_state
    print(f"State: {state.value:12} (confidence: {confidence:.2f}, {frames} frames)")


if __name__ == "__main__":
    # Test state machine
    sm = WorkZoneStateMachine()
    print(f"\n✓ State Machine initialized")
    print(f"  - Initial state: {sm.current_state.value}")
    print(f"  - Transitions config loaded")
