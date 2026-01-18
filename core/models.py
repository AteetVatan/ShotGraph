"""Pydantic models for ShotGraph domain entities."""

from enum import Enum

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages for TTS and subtitles."""

    ENGLISH = "en"
    HINDI = "hi"


class JobStatus(str, Enum):
    """Status of a video generation job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ShotType(str, Enum):
    """Types of camera shots."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    ESTABLISHING = "establishing"


class StoryEntities(BaseModel):
    """Extracted entities from a story."""

    characters: list[str] = Field(
        default_factory=list,
        description="Main characters identified in the story",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Locations/settings identified in the story",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Themes or key concepts in the story",
    )
    organizations: list[str] = Field(
        default_factory=list,
        description="Organizations or groups mentioned",
    )


class ProcessedStory(BaseModel):
    """Preprocessed story with NLP enhancements."""

    original_text: str = Field(..., description="Original unprocessed story text")
    chunks: list[str] = Field(
        default_factory=list,
        description="Story split into LLM-friendly chunks",
    )
    summary: str | None = Field(
        None,
        description="Condensed summary for context reduction",
    )
    entities: StoryEntities = Field(
        default_factory=StoryEntities,
        description="Extracted named entities",
    )
    token_count: int = Field(
        default=0,
        description="Estimated token count of original text",
    )
    was_chunked: bool = Field(
        default=False,
        description="Whether the story was split into chunks",
    )


class StoryInput(BaseModel):
    """Input model for story text submission."""

    text: str = Field(..., min_length=10, description="The story text to convert to video")
    title: str | None = Field(None, description="Optional title for the story")
    language: Language = Field(Language.ENGLISH, description="Primary language of the story")


class Shot(BaseModel):
    """A single shot within a scene."""

    id: int = Field(..., description="Shot ID within the scene")
    scene_id: int = Field(..., description="Parent scene ID")
    description: str = Field(..., description="Visual description for video generation")
    duration_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Duration of the shot in seconds",
    )
    shot_type: ShotType | None = Field(None, description="Type of camera shot")
    dialogue: str | None = Field(None, description="Dialogue text for TTS")
    subtitle_text: str | None = Field(None, description="Subtitle text to display")
    video_file_path: str | None = Field(None, description="Path to generated video file")
    audio_file_path: str | None = Field(None, description="Path to generated audio file")


class Scene(BaseModel):
    """A scene containing multiple shots."""

    id: int = Field(..., description="Scene ID")
    text: str = Field(..., description="Original story text for this scene")
    summary: str = Field(..., description="Brief summary of the scene")
    shots: list[Shot] = Field(default_factory=list, description="Shots in this scene")
    music_file_path: str | None = Field(None, description="Path to background music file")


class SceneList(BaseModel):
    """Collection of scenes parsed from a story."""

    scenes: list[Scene] = Field(..., description="List of scenes")


class VideoJob(BaseModel):
    """Represents a video generation job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(JobStatus.PENDING, description="Current job status")
    story_input: StoryInput = Field(..., description="Original story input")
    processed_story: ProcessedStory | None = Field(
        None,
        description="NLP-processed story with entities and chunks",
    )
    scenes: SceneList | None = Field(None, description="Parsed scenes")
    final_video_path: str | None = Field(None, description="Path to final video")
    error_message: str | None = Field(None, description="Error message if failed")
    progress: str | None = Field(None, description="Current progress description")
    current_stage: str | None = Field(None, description="Current pipeline stage")
