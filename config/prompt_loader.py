"""Prompt loading utility for reading prompt templates from files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants for prompt file names
PROMPT_SCENE_BREAKDOWN = "scene_breakdown"
PROMPT_SHOT_PLANNING = "shot_planning"
PROMPT_STYLE_GUIDE = "style_guide"

# TOON format variants
PROMPT_SCENE_BREAKDOWN_TOON = "scene_breakdown_toon"
PROMPT_SHOT_PLANNING_TOON = "shot_planning_toon"

# Base directory for prompts
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(prompt_name: str, *, use_toon: bool = False) -> str:
    """Load prompt text from a file in the prompts directory.

    Args:
        prompt_name: Name of the prompt file (without .txt extension).
        use_toon: If True, attempt to load TOON variant first.

    Returns:
        The prompt text content.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        IOError: If the file cannot be read.
    """
    # Try TOON variant if requested
    if use_toon:
        toon_name = f"{prompt_name}_toon"
        toon_path = _PROMPTS_DIR / f"{toon_name}.txt"
        if toon_path.exists():
            try:
                content = toon_path.read_text(encoding="utf-8")
                logger.debug("Loaded TOON prompt from %s (%d chars)", toon_path, len(content))
                return content.strip()
            except IOError:
                logger.warning("Failed to read TOON prompt %s, falling back to standard", toon_path)

    prompt_path = _PROMPTS_DIR / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. "
            f"Expected location: {_PROMPTS_DIR}/"
        )

    try:
        content = prompt_path.read_text(encoding="utf-8")
        logger.debug("Loaded prompt from %s (%d chars)", prompt_path, len(content))
        return content.strip()
    except IOError as e:
        raise IOError(f"Failed to read prompt file {prompt_path}: {e}") from e


def get_prompt_for_format(base_name: str, *, use_toon: bool = False) -> str:
    """Get the appropriate prompt variant based on format setting.

    Args:
        base_name: Base name of the prompt (e.g., "scene_breakdown").
        use_toon: Whether to use TOON format.

    Returns:
        The prompt content.
    """
    return load_prompt(base_name, use_toon=use_toon)
