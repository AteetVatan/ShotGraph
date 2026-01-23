"""TTS audio generation agent."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.models import Shot
from core.protocols.tts_generator import ITTSGenerator

from .base import BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TTSAgent(BaseAgent[Shot, Path]):
    """Agent that generates TTS audio for shot dialogue.

    Uses the injected TTS generator service to create speech audio
    from dialogue or subtitle text in shots.
    """

    def __init__(
        self,
        *,
        tts_generator: ITTSGenerator,
        max_retries: int = 2,
    ):
        """Initialize the TTS agent.

        Args:
            tts_generator: Service for generating TTS audio.
            max_retries: Maximum retry attempts.
        """
        super().__init__(max_retries=max_retries)
        self._generator = tts_generator

    async def _execute(self, shot: Shot) -> Path:
        """Generate TTS audio for the shot.

        Args:
            shot: The shot to generate audio for.

        Returns:
            Path to the generated audio file.

        Raises:
            ValueError: If shot has no dialogue or subtitle text.
        """
        # Get text to speak
        text = shot.dialogue or shot.subtitle_text
        if not text:
            raise ValueError(f"Shot {shot.id} has no dialogue or subtitle text")

        self.logger.info(
            "Generating TTS for scene %d, shot %d: %s...",
            shot.scene_id,
            shot.id,
            text[:30],
        )

        from core.models import Language

        audio_path = await self._generator.generate(
            text=text,
            language=Language.ENGLISH,  # Default to English
        )

        self.logger.info("Generated TTS audio: %s", audio_path)
        return audio_path
