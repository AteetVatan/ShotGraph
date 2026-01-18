"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field, field_validator

from core.models import JobStatus, Language

# Default limits (can be overridden by settings at runtime)
DEFAULT_MAX_STORY_LENGTH = 100_000
DEFAULT_MAX_STORY_SIZE_BYTES = 500_000


class GenerateRequest(BaseModel):
    """Request body for video generation endpoint."""

    story: str = Field(
        ...,
        min_length=50,
        max_length=DEFAULT_MAX_STORY_LENGTH,
        description="The story text to convert to video (50-100,000 characters)",
        examples=["Once upon a time in a small village, there lived a young hero..."],
    )
    title: str | None = Field(
        None,
        max_length=200,
        description="Optional title for the video (max 200 characters)",
        examples=["The Hero's Journey"],
    )
    language: Language = Field(
        Language.ENGLISH,
        description="Primary language for TTS and subtitles",
    )

    @field_validator("story")
    @classmethod
    def validate_story_size(cls, v: str) -> str:
        """Validate story size in bytes to prevent oversized requests.

        Args:
            v: The story text.

        Returns:
            The validated story text.

        Raises:
            ValueError: If story exceeds maximum size.
        """
        size_bytes = len(v.encode("utf-8"))
        if size_bytes > DEFAULT_MAX_STORY_SIZE_BYTES:
            raise ValueError(
                f"Story exceeds maximum size of {DEFAULT_MAX_STORY_SIZE_BYTES // 1000}KB "
                f"(got {size_bytes // 1000}KB)"
            )
        return v

    @field_validator("story")
    @classmethod
    def validate_story_content(cls, v: str) -> str:
        """Basic content validation for story text.

        Args:
            v: The story text.

        Returns:
            The validated story text.

        Raises:
            ValueError: If story contains only whitespace.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Story cannot be empty or contain only whitespace")
        if len(stripped.split()) < 10:
            raise ValueError("Story must contain at least 10 words")
        return v


class GenerateResponse(BaseModel):
    """Response from video generation endpoint."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")


class StatusResponse(BaseModel):
    """Response from job status endpoint."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current job status")
    progress: str | None = Field(None, description="Current progress description")
    current_stage: str | None = Field(None, description="Current pipeline stage")
    final_video_path: str | None = Field(None, description="Path to completed video")
    error_message: str | None = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    profile: str = Field(..., description="Current execution profile")


class JobListResponse(BaseModel):
    """Response listing all jobs."""

    jobs: list[StatusResponse] = Field(..., description="List of all jobs")
    total: int = Field(..., description="Total number of jobs")
