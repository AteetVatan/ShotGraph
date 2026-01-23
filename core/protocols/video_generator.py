"""Protocol for video generator interface."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class VideoGenerationResult:
    """Result of video generation containing path and actual duration.

    Attributes:
        path: Path to the generated video file.
        actual_duration: The actual duration of the generated video in seconds.
            This may differ from the requested duration due to frame limits.
    """

    path: Path
    actual_duration: float


class IVideoGenerator(Protocol):
    """Interface for video generation services."""

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        init_image: Path | None = None,
        seed: int | None = None,
    ) -> VideoGenerationResult:
        """Generate a video clip from a text prompt.

        Args:
            prompt: Visual description for video generation.
            duration_seconds: Desired duration of the clip.
            init_image: Optional initial frame for continuity.
            seed: Optional random seed for reproducibility.

        Returns:
            VideoGenerationResult containing path and actual duration.

        Raises:
            VideoGenerationError: If generation fails.
        """
        ...
