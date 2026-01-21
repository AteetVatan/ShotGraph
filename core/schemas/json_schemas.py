"""JSON schemas for structured outputs and validation."""

from typing import Any

from core.models import Scene, SceneList, Shot


def get_scene_list_schema() -> dict[str, Any]:
    """Get JSON schema for SceneList model.

    Returns:
        JSON schema dictionary compatible with Together.ai structured outputs.
    """
    return {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "summary": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "summary", "text"],
                },
            }
        },
        "required": ["scenes"],
    }


def get_shot_list_schema() -> dict[str, Any]:
    """Get JSON schema for list of Shot models.

    Returns:
        JSON schema dictionary compatible with Together.ai structured outputs.
    """
    return {
        "type": "object",
        "properties": {
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "description": {"type": "string"},
                        "duration": {"type": "number", "minimum": 1.0, "maximum": 30.0},
                        "shot_type": {
                            "type": "string",
                            "enum": ["wide", "medium", "close_up", "close-up", "closeup", "establishing", None],
                        },
                        "dialogue": {"type": ["string", "null"]},
                    },
                    "required": ["id", "description"],
                },
            }
        },
        "required": ["shots"],
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

    scenes_data = data.get("scenes", [])
    if not scenes_data:
        raise ValueError("No scenes found in data")

    scenes = []
    for scene_data in scenes_data:
        scene = Scene(
            id=scene_data["id"],
            summary=scene_data["summary"],
            text=scene_data["text"],
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
    from core.models import Shot, ShotType

    shots_data = data.get("shots", [])
    if not shots_data:
        raise ValueError("No shots found in data")

    shots = []
    for shot_data in shots_data:
        # Map shot_type string to enum
        shot_type_str = shot_data.get("shot_type")
        shot_type = None
        if shot_type_str:
            mapping = {
                "wide": ShotType.WIDE,
                "medium": ShotType.MEDIUM,
                "close_up": ShotType.CLOSE_UP,
                "close-up": ShotType.CLOSE_UP,
                "closeup": ShotType.CLOSE_UP,
                "establishing": ShotType.ESTABLISHING,
            }
            shot_type = mapping.get(shot_type_str.lower())

        # Handle null dialogue
        dialogue = shot_data.get("dialogue")
        if dialogue in ("null", "None", None):
            dialogue = None

        shot = Shot(
            id=shot_data["id"],
            scene_id=scene_id,
            description=shot_data["description"],
            duration_seconds=float(shot_data.get("duration", 5.0)),
            shot_type=shot_type,
            dialogue=dialogue,
            subtitle_text=dialogue,
        )
        shots.append(shot)

    return shots
