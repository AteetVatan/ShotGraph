"""Protocol for TTS generator interface."""

from pathlib import Path
from typing import Protocol

from core.models import Language


class ITTSGenerator(Protocol):
    """Interface for Text-to-Speech generation services."""

    async def generate(
        self,
        *,
        text: str,
        language: Language = Language.ENGLISH,
        output_path: Path | None = None,
    ) -> Path:
        """Generate speech audio from text.

        Args:
            text: The text to convert to speech.
            language: Target language for TTS.
            output_path: Optional specific output path.

        Returns:
            Path to the generated audio file.

        Raises:
            TTSGenerationError: If generation fails.
        """
        ...
