"""Prompt loading utility for reading prompt templates from files."""

import logging
import re
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Base directory for prompts
_PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptName(str, Enum):
    """Enumeration of all available prompt names for type safety."""

    # Scene and shot planning
    SCENE_BREAKDOWN = "scene_breakdown"
    SCENE_BREAKDOWN_TOON = "scene_breakdown_toon"
    SHOT_PLANNING = "shot_planning"
    SHOT_PLANNING_TOON = "shot_planning_toon"
    STYLE_GUIDE = "style_guide"

    # Story processing
    STORY_SUMMARIZATION_SYSTEM = "story_summarization_system"
    STORY_SUMMARIZATION_USER = "story_summarization_user"

    # JSON repair
    JSON_REPAIR_SYSTEM = "json_repair_system"
    JSON_REPAIR_USER = "json_repair_user"

    # Content moderation
    CONTENT_MODERATION_SYSTEM = "content_moderation_system"
    CONTENT_MODERATION_USER = "content_moderation_user"


# Backward compatibility constants
PROMPT_SCENE_BREAKDOWN = PromptName.SCENE_BREAKDOWN.value
PROMPT_SHOT_PLANNING = PromptName.SHOT_PLANNING.value
PROMPT_STYLE_GUIDE = PromptName.STYLE_GUIDE.value
PROMPT_SCENE_BREAKDOWN_TOON = PromptName.SCENE_BREAKDOWN_TOON.value
PROMPT_SHOT_PLANNING_TOON = PromptName.SHOT_PLANNING_TOON.value


def _extract_template_vars(template: str) -> set[str]:
    """Extract variable names from template string.

    Args:
        template: Template string with {var_name} placeholders.

    Returns:
        Set of variable names found in template.
    """
    return set(re.findall(r"\{(\w+)\}", template))


def _substitute_template(template: str, **vars: str) -> str:
    """Substitute variables in template string.

    Args:
        template: Template string with {var_name} placeholders.
        **vars: Variables to substitute.

    Returns:
        Template with variables substituted.

    Raises:
        KeyError: If a template variable is missing from vars.
    """
    if not vars:
        return template

    try:
        return template.format(**vars)
    except KeyError as e:
        required = _extract_template_vars(template)
        raise KeyError(
            f"Missing template variable: {e}. "
            f"Required variables: {sorted(required)}"
        ) from e


def load_prompt(
    prompt_name: str | PromptName,
    *,
    use_toon: bool = False,
    **template_vars: str,
) -> str:
    """Load prompt text from a file with optional template variable substitution.

    Args:
        prompt_name: Name of the prompt file (without .txt extension) or PromptName enum.
        use_toon: If True, attempt to load TOON variant first.
        **template_vars: Variables to substitute in the template using {var_name} syntax.

    Returns:
        The prompt text content with variables substituted.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        IOError: If the file cannot be read.
        KeyError: If a template variable is missing.
    """
    # Convert enum to string if needed
    if isinstance(prompt_name, PromptName):
        prompt_name = prompt_name.value

    # Try TOON variant if requested
    if use_toon:
        toon_name = f"{prompt_name}_toon"
        toon_path = _PROMPTS_DIR / f"{toon_name}.txt"
        if toon_path.exists():
            try:
                content = toon_path.read_text(encoding="utf-8")
                logger.debug("Loaded TOON prompt from %s (%d chars)", toon_path, len(content))
                return _substitute_template(content.strip(), **template_vars)
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
        return _substitute_template(content.strip(), **template_vars)
    except IOError as e:
        raise IOError(f"Failed to read prompt file {prompt_path}: {e}") from e


def get_prompt_for_format(base_name: str | PromptName, *, use_toon: bool = False) -> str:
    """Get the appropriate prompt variant based on format setting.

    Args:
        base_name: Base name of the prompt (e.g., "scene_breakdown").
        use_toon: Whether to use TOON format.

    Returns:
        The prompt content.
    """
    return load_prompt(base_name, use_toon=use_toon)
