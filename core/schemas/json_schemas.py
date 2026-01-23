"""JSON schemas for structured outputs and validation."""

from typing import Any

from core.constants import FieldNames, NullValues, SHOT_TYPE_MAPPING
from core.models import Scene, SceneList, Shot, ShotType


def get_scene_list_schema() -> dict[str, Any]:
    """Get JSON schema for SceneList model.

    Returns:
        JSON schema dictionary compatible with Together.ai structured outputs.
    """
    return {
        "type": "object",
        "properties": {
            FieldNames.SCENES: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        FieldNames.ID: {"type": "integer"},
                        FieldNames.SUMMARY: {"type": "string"},
                        FieldNames.TEXT: {"type": "string"},
                    },
                    "required": [FieldNames.ID, FieldNames.SUMMARY, FieldNames.TEXT],
                },
            }
        },
        "required": [FieldNames.SCENES],
    }


def get_shot_list_schema() -> dict[str, Any]:
    """Get JSON schema for list of Shot models.

    Returns:
        JSON schema dictionary compatible with Together.ai structured outputs.
    """
    # Build enum list from mapping keys
    shot_type_enum = list(SHOT_TYPE_MAPPING.keys()) + [None]
    
    return {
        "type": "object",
        "properties": {
            FieldNames.SHOTS: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        FieldNames.ID: {"type": "integer"},
                        FieldNames.DESCRIPTION: {"type": "string"},
                        FieldNames.DURATION: {"type": "number", "minimum": 1.0, "maximum": 30.0},
                        FieldNames.SHOT_TYPE: {
                            "type": "string",
                            "enum": shot_type_enum,
                        },
                        FieldNames.DIALOGUE: {"type": ["string", "null"]},
                    },
                    "required": [FieldNames.ID, FieldNames.DESCRIPTION],
                },
            }
        },
        "required": [FieldNames.SHOTS],
    }


def validate_scene_list(data: dict[str, Any]) -> SceneList:
    """Validate and create SceneList from dictionary.

    Args:
        data: Dictionary parsed from JSON/TOON.

    Returns:
        Validated SceneList model.

    Raises:
        ValueError: If data doesn't match schema.
    """
    from core.models import Scene

    scenes_data = data.get(FieldNames.SCENES, [])
    if not scenes_data:
        raise ValueError("No scenes found in data")

    scenes = []
    for scene_data in scenes_data:
        scene = Scene(
            id=scene_data[FieldNames.ID],
            summary=scene_data[FieldNames.SUMMARY],
            text=scene_data[FieldNames.TEXT],
        )
        scenes.append(scene)

    return SceneList(scenes=scenes)


def validate_shot_list(data: dict[str, Any], scene_id: int) -> list[Shot]:
    """Validate and create list of Shot from dictionary.

    Args:
        data: Dictionary parsed from JSON/TOON.
        scene_id: Parent scene ID for shots.

    Returns:
        Validated list of Shot models.

    Raises:
        ValueError: If data doesn't match schema.
    """
    from core.models import Shot

    shots_data = data.get(FieldNames.SHOTS, [])
    if not shots_data:
        raise ValueError("No shots found in data")

    shots = []
    for shot_data in shots_data:
        # Map shot_type string to enum
        shot_type_str = shot_data.get(FieldNames.SHOT_TYPE)
        shot_type = None
        if shot_type_str:
            shot_type = SHOT_TYPE_MAPPING.get(shot_type_str.lower())

        # Handle null dialogue
        dialogue = shot_data.get(FieldNames.DIALOGUE)
        if dialogue in (NullValues.NULL, NullValues.NONE, None):
            dialogue = None

        shot = Shot(
            id=shot_data[FieldNames.ID],
            scene_id=scene_id,
            description=shot_data[FieldNames.DESCRIPTION],
            duration_seconds=float(shot_data.get(FieldNames.DURATION, 5.0)),
            shot_type=shot_type,
            dialogue=dialogue,
            subtitle_text=dialogue,
        )
        shots.append(shot)

    return shots
