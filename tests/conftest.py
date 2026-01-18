"""Pytest configuration and shared fixtures."""

import pytest

from config.settings import ExecutionProfile, Settings
from core.models import Language, Scene, SceneList, Shot, ShotType, StoryInput


@pytest.fixture
def test_settings() -> Settings:
    """Provide test settings with debug profile."""
    return Settings(
        execution_profile=ExecutionProfile.DEBUG_CPU,
        gpu_enabled=False,
        llm_api_key="test_key",
        storage_path="./test_output",
        assets_path="./test_assets",
        max_retries=1,
    )


@pytest.fixture
def sample_story_input() -> StoryInput:
    """Provide a sample story input for testing."""
    return StoryInput(
        text="""Once upon a time in a small village, there lived a young hero named Alex.
        Alex dreamed of adventure beyond the mountains.
        
        One morning, Alex met an old wizard at the village well.
        The wizard spoke of a dragon threatening the kingdom.
        
        Alex decided to embark on a quest to defeat the dragon.
        The journey would take them through dark forests and across raging rivers.""",
        title="The Hero's Quest",
        language=Language.ENGLISH,
    )


@pytest.fixture
def sample_scene() -> Scene:
    """Provide a sample scene for testing."""
    return Scene(
        id=1,
        text="Alex met an old wizard at the village well. The wizard spoke of a dragon.",
        summary="Hero meets the wizard who reveals the dragon threat",
    )


@pytest.fixture
def sample_shot() -> Shot:
    """Provide a sample shot for testing."""
    return Shot(
        id=1,
        scene_id=1,
        description="A wide shot of a medieval village well with an elderly wizard in blue robes",
        duration_seconds=5.0,
        shot_type=ShotType.WIDE,
        dialogue="There is a dragon threatening our kingdom.",
    )


@pytest.fixture
def sample_scene_list(sample_scene: Scene) -> SceneList:
    """Provide a sample scene list for testing."""
    return SceneList(scenes=[sample_scene])
