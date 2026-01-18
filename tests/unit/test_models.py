"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from core.models import (
    JobStatus,
    Language,
    Scene,
    SceneList,
    Shot,
    ShotType,
    StoryInput,
    VideoJob,
)


class TestStoryInput:
    """Tests for StoryInput model."""

    def test_valid_story_input(self) -> None:
        """Test creating a valid story input."""
        story = StoryInput(text="A long story about heroes and adventures.")
        assert story.language == Language.ENGLISH
        assert story.title is None

    def test_story_input_with_all_fields(self) -> None:
        """Test story input with all fields populated."""
        story = StoryInput(
            text="A Hindi story about friendship.",
            title="Friendship Tales",
            language=Language.HINDI,
        )
        assert story.title == "Friendship Tales"
        assert story.language == Language.HINDI

    def test_story_input_text_too_short(self) -> None:
        """Test that short text is rejected."""
        with pytest.raises(ValidationError):
            StoryInput(text="Short")


class TestShot:
    """Tests for Shot model."""

    def test_valid_shot(self) -> None:
        """Test creating a valid shot."""
        shot = Shot(
            id=1,
            scene_id=1,
            description="A wide shot of the village",
        )
        assert shot.duration_seconds == 5.0
        assert shot.shot_type is None
        assert shot.video_file_path is None

    def test_shot_with_all_fields(self) -> None:
        """Test shot with all optional fields."""
        shot = Shot(
            id=2,
            scene_id=1,
            description="Close-up of the hero's face",
            duration_seconds=7.5,
            shot_type=ShotType.CLOSE_UP,
            dialogue="I will save the kingdom!",
            subtitle_text="I will save the kingdom!",
            video_file_path="/output/shot_2.mp4",
            audio_file_path="/output/shot_2.wav",
        )
        assert shot.shot_type == ShotType.CLOSE_UP
        assert shot.duration_seconds == 7.5

    def test_shot_duration_too_short(self) -> None:
        """Test that duration < 1 second is rejected."""
        with pytest.raises(ValidationError):
            Shot(id=1, scene_id=1, description="test", duration_seconds=0.5)

    def test_shot_duration_too_long(self) -> None:
        """Test that duration > 30 seconds is rejected."""
        with pytest.raises(ValidationError):
            Shot(id=1, scene_id=1, description="test", duration_seconds=35.0)


class TestScene:
    """Tests for Scene model."""

    def test_valid_scene(self) -> None:
        """Test creating a valid scene."""
        scene = Scene(
            id=1,
            text="The hero enters the castle.",
            summary="Hero arrives at the castle",
        )
        assert scene.shots == []
        assert scene.music_file_path is None

    def test_scene_with_shots(self) -> None:
        """Test scene with populated shots list."""
        shots = [
            Shot(id=1, scene_id=1, description="Wide shot of castle"),
            Shot(id=2, scene_id=1, description="Hero walking toward gate"),
        ]
        scene = Scene(
            id=1,
            text="The hero approaches the castle.",
            summary="Castle approach scene",
            shots=shots,
        )
        assert len(scene.shots) == 2


class TestSceneList:
    """Tests for SceneList model."""

    def test_valid_scene_list(self) -> None:
        """Test creating a valid scene list."""
        scenes = [
            Scene(id=1, text="Scene 1 text", summary="Scene 1"),
            Scene(id=2, text="Scene 2 text", summary="Scene 2"),
        ]
        scene_list = SceneList(scenes=scenes)
        assert len(scene_list.scenes) == 2


class TestVideoJob:
    """Tests for VideoJob model."""

    def test_valid_job(self) -> None:
        """Test creating a valid video job."""
        story = StoryInput(text="A test story for the video job.")
        job = VideoJob(job_id="test-123", story_input=story)
        assert job.status == JobStatus.PENDING
        assert job.scenes is None
        assert job.final_video_path is None

    def test_job_status_progression(self) -> None:
        """Test job status can be updated."""
        story = StoryInput(text="Another test story for status check.")
        job = VideoJob(job_id="test-456", story_input=story)
        
        job.status = JobStatus.PROCESSING
        assert job.status == JobStatus.PROCESSING
        
        job.status = JobStatus.COMPLETED
        assert job.status == JobStatus.COMPLETED
