"""Protocol for music generator interface."""

from pathlib import Path
from typing import Protocol


class IMusicGenerator(Protocol):
    """Interface for music generation services."""

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        output_path: Path | None = None,
    ) -> Path:
        """Generate background music from a text prompt.

        Args:
            prompt: Description of the desired music style/mood.
            duration_seconds: Desired duration of the music.
            output_path: Optional specific output path.

        Returns:
            Path to the generated audio file.

        Raises:
            MusicGenerationError: If generation fails.
        """
        ...
