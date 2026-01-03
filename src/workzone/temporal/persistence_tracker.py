"""
Persistence Tracker - Tracks cue presence over time
Phase 1.1: Multi-Cue AND + Temporal Persistence
"""

from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
import numpy as np
import yaml
from pathlib import Path

from ..detection import FrameCues


@dataclass
class PersistenceState:
    """Persistence state for a single cue group."""
    cue_group: str
    history: deque  # Boolean history of presence
    persistence_score: float  # Fraction of frames present
    is_sustained: bool  # Above persistence threshold
    frames_sustained: int  # Consecutive frames above threshold


class PersistenceTracker:
    """
    Tracks cue presence over temporal sliding window.
    
    Maintains a history buffer (deque) per cue group and computes:
    - Persistence score: fraction of frames cue was present
    - Sustained status: whether persistence exceeds threshold
    - Temporal trends: increasing/decreasing presence
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize persistence tracker.
        
        Args:
            config_path: Path to multi_cue_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "multi_cue_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get temporal parameters
        self.window_size = self.config['temporal']['window_size']
        self.persistence_threshold = self.config['temporal']['persistence_threshold']
        
        # Get cue groups
        self.cue_groups = list(self.config['cue_groups'].keys())
        
        # Initialize history buffers
        self.history: Dict[str, deque] = {
            group: deque(maxlen=self.window_size)
            for group in self.cue_groups
        }
        
        # Track sustained frames counter
        self.sustained_counter: Dict[str, int] = {
            group: 0 for group in self.cue_groups
        }
        
        # Frame counter
        self.frame_count = 0
    
    def update(self, frame_cues: FrameCues) -> Dict[str, PersistenceState]:
        """
        Update persistence tracker with new frame cues.
        
        Args:
            frame_cues: FrameCues object from CueClassifier
            
        Returns:
            Dictionary of PersistenceState per cue group
        """
        self.frame_count += 1
        persistence_states = {}
        
        for group in self.cue_groups:
            # Get cue presence for this frame
            is_present = frame_cues.cue_groups[group]['present']
            
            # Add to history
            self.history[group].append(is_present)
            
            # Compute persistence score
            if len(self.history[group]) == 0:
                persistence_score = 0.0
            else:
                persistence_score = sum(self.history[group]) / len(self.history[group])
            
            # Check if sustained
            is_sustained = persistence_score >= self.persistence_threshold
            
            # Update sustained counter
            if is_sustained:
                self.sustained_counter[group] += 1
            else:
                self.sustained_counter[group] = 0
            
            # Create persistence state
            persistence_states[group] = PersistenceState(
                cue_group=group,
                history=self.history[group].copy(),
                persistence_score=persistence_score,
                is_sustained=is_sustained,
                frames_sustained=self.sustained_counter[group]
            )
        
        return persistence_states
    
    def get_sustained_cues(self, persistence_states: Dict[str, PersistenceState]) -> List[str]:
        """
        Get list of cue groups that are currently sustained.
        
        Args:
            persistence_states: Dictionary from update()
            
        Returns:
            List of cue group names that meet persistence threshold
        """
        return [
            group for group, state in persistence_states.items()
            if state.is_sustained
        ]
    
    def get_summary(self, persistence_states: Dict[str, PersistenceState]) -> Dict:
        """
        Get summary statistics of persistence state.
        
        Returns:
            Dictionary with persistence statistics
        """
        sustained_cues = self.get_sustained_cues(persistence_states)
        
        return {
            'frame_count': self.frame_count,
            'num_sustained_cues': len(sustained_cues),
            'sustained_cues': sustained_cues,
            'persistence_scores': {
                group: state.persistence_score
                for group, state in persistence_states.items()
            },
            'max_persistence': max(
                (state.persistence_score for state in persistence_states.values()),
                default=0.0
            ),
            'min_persistence': min(
                (state.persistence_score for state in persistence_states.values()),
                default=0.0
            )
        }
    
    def reset(self):
        """Reset persistence tracker to initial state."""
        for group in self.cue_groups:
            self.history[group].clear()
            self.sustained_counter[group] = 0
        self.frame_count = 0
    
    def get_trend(self, cue_group: str, window: int = 10) -> str:
        """
        Get trend direction for a cue group (increasing/decreasing/stable).
        
        Args:
            cue_group: Cue group name
            window: Number of recent frames to analyze
            
        Returns:
            'increasing', 'decreasing', or 'stable'
        """
        history = list(self.history[cue_group])
        if len(history) < window:
            return 'stable'
        
        recent = history[-window:]
        earlier = history[-2*window:-window] if len(history) >= 2*window else history[:-window]
        
        if len(earlier) == 0:
            return 'stable'
        
        recent_score = sum(recent) / len(recent)
        earlier_score = sum(earlier) / len(earlier)
        
        diff = recent_score - earlier_score
        if diff > 0.2:
            return 'increasing'
        elif diff < -0.2:
            return 'decreasing'
        else:
            return 'stable'


# ===================================
# UTILITY FUNCTIONS
# ===================================

def print_persistence_state(persistence_states: Dict[str, PersistenceState]):
    """Print persistence state for debugging."""
    print("\n=== PERSISTENCE STATE ===")
    for group, state in persistence_states.items():
        status = "✓ SUSTAINED" if state.is_sustained else "✗ not sustained"
        print(f"{group:20} {state.persistence_score:.2f} {status:15} ({state.frames_sustained} frames)")


if __name__ == "__main__":
    # Test persistence tracker
    from ..detection import CueClassifier, FrameCues
    
    tracker = PersistenceTracker()
    print(f"\n✓ Persistence Tracker initialized")
    print(f"  - Window size: {tracker.window_size} frames")
    print(f"  - Persistence threshold: {tracker.persistence_threshold}")
    print(f"  - Tracking {len(tracker.cue_groups)} cue groups")
