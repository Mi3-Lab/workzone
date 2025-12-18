"""Global project configuration and entry points."""

import sys
from pathlib import Path

# Add src to path for imports
SRC_PATH = Path(__file__).parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
