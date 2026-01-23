"""Constants for field names, null values, and schema types.

This module centralizes all hardcoded string literals used as dictionary keys,
field names, and special values throughout the application to improve maintainability
and reduce typos.
"""

from enum import Enum
from typing import Final

from core.models import ShotType


class FieldNames:
    """Field names used in dictionaries and JSON/TOON parsing."""

    # Shot fields
    SHOT_TYPE: Final[str] = "shot_type"
    DIALOGUE: Final[str] = "dialogue"
    SUBTITLE_TEXT: Final[str] = "subtitle_text"
    DESCRIPTION: Final[str] = "description"
    DURATION: Final[str] = "duration"
    VISUAL_STYLE: Final[str] = "visual_style"

    # Scene fields
    SCENE_ID: Final[str] = "scene_id"
    SUMMARY: Final[str] = "summary"
    TEXT: Final[str] = "text"

    # Common fields
    ID: Final[str] = "id"

    # Collection fields
    SHOTS: Final[str] = "shots"
    SCENES: Final[str] = "scenes"

    # Style context fields
    SETTING: Final[str] = "setting"
    TIME_OF_DAY: Final[str] = "time_of_day"
    WEATHER: Final[str] = "weather"
    MOOD: Final[str] = "mood"


class NullValues:
    """String representations of null/None values."""

    NULL: Final[str] = "null"
    NONE: Final[str] = "None"


class SchemaType(str, Enum):
    """Schema type identifiers for JSON repair and validation."""

    SCENE_LIST = "scene_list"
    SHOT_LIST = "shot_list"


# Shot type string mappings to enum
# Maps various string representations to ShotType enum values
SHOT_TYPE_MAPPING: Final[dict[str, ShotType]] = {
    "wide": ShotType.WIDE,
    "medium": ShotType.MEDIUM,
    "close_up": ShotType.CLOSE_UP,
    "close-up": ShotType.CLOSE_UP,
    "closeup": ShotType.CLOSE_UP,
    "establishing": ShotType.ESTABLISHING,
}


# Video frame safety margins
# Used to avoid reading corrupt/incomplete last frames from AI-generated videos
FRAME_SAFETY_MIN_OFFSET: Final[float] = 0.05  # minimum 50ms from boundary
FRAME_SAFETY_FRAME_COUNT: Final[int] = 2  # at least N frames back from boundary