"""Protocol for video generator interface."""

from pathlib import Path
from typing import Protocol


class IVideoGenerator(Protocol):
    """Interface for video generation services."""

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        init_image: Path | None = None,
        seed: int | None = None,
    ) -> Path:
        """Generate a video clip from a text prompt.

        Args:
            prompt: Visual description for video generation.
            duration_seconds: Desired duration of the clip.
            init_image: Optional initial frame for continuity.
            seed: Optional random seed for reproducibility.

        Returns:
            Path to the generated video file.

        Raises:
            VideoGenerationError: If generation fails.
        """
        ...
