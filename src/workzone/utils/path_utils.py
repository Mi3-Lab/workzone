"""Path and file utilities."""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object pointing to the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
