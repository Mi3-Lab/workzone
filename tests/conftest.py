"""Test configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Get test data directory."""
    return project_root / "data"


@pytest.fixture
def weights_dir(project_root: Path) -> Path:
    """Get weights directory."""
    return project_root / "weights"
