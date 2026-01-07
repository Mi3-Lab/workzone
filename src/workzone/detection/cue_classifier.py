from typing import Optional

from workzone.apps.streamlit_utils import (
    CHANNELIZATION,
    WORKERS,
    VEHICLES,
    MESSAGE_BOARD,
    OTHER_ROADWORK,
    is_ttc_sign,
)


class CueClassifier:
    """Simple cue classifier for Phase 2.1 per-cue verification."""
    
    def map_name_to_cue(self, name: str) -> Optional[str]:
        """Map YOLO class name to cue category."""
        n = name.strip()
        if n in CHANNELIZATION:
            return "channelization"
        if n in WORKERS:
            return "workers"
        if n in VEHICLES:
            return "vehicles"
        if is_ttc_sign(n) or n in MESSAGE_BOARD:
            return "signs"
        if n in OTHER_ROADWORK:
            return "equipment"
        return None
