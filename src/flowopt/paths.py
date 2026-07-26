"""Locations of the repo's non-code assets.

`config/`, `prompts/` and `skills/` are edited far more often than the code, so
they live at the repo root rather than inside the package. That means the package
has to find the root: it is two levels up from this file, which holds for a source
checkout and for an editable install (`uv sync`).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
SKILLS_DIR = ROOT / "skills"


def resolve(path) -> Path:
    """Make a config-supplied path absolute, relative to the repo root.

    Config paths have to survive being read from somewhere other than the repo
    root: the design agent runs in a scratch directory, and an experiment script
    runs from its own folder. Relative-to-root is the only reading that means the
    same thing in all three places.

    Args:
        path: A path from config — absolute, or relative to the repo root.

    Returns:
        The absolute Path. An already-absolute input is returned unchanged.
    """
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
