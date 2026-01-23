"""Text-to-speech service implementations."""

import hashlib
import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import TTSGenerationError
from core.models import Language

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class MockTTSGenerator:
    """Mock TTS generator for DEBUG_CPU mode.

    Generates silent audio files of appropriate duration for testing.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the mock TTS generator.

        Args:
            settings: Application settings.
        """
        self._output_dir = Path(settings.storage_path) / "mock_audio"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Path] = {}

    async def generate(
        self,
        *,
        text: str,
        language: Language = Language.ENGLISH,
        output_path: Path | None = None,
    ) -> Path:
        """Generate a silent audio file for the text duration.

        Args:
            text: The text to "speak" (used to estimate duration).
            language: Target language (affects duration estimation).
            output_path: Optional specific output path.

        Returns:
            Path to the generated audio file.
        """
        # Estimate duration based on text length (rough: 150 words/minute)
        word_count = len(text.split())
        duration_seconds = max(1.0, word_count / 2.5)  # ~2.5 words/second

        cache_key = self._get_cache_key(text, language)
        if cache_key in self._cache and self._cache[cache_key].exists():
            logger.debug("Using cached mock audio: %s", cache_key)
            return self._cache[cache_key]

        logger.info("Generating mock TTS audio (%.1fs) for: %s...", duration_seconds, text[:30])

        try:
            if output_path:
                audio_path = output_path
            else:
                audio_path = self._output_dir / f"{cache_key}.wav"

            # Generate silent WAV file
            self._generate_silent_wav(audio_path, duration_seconds)

            self._cache[cache_key] = audio_path
            logger.info("Generated mock TTS: %s", audio_path)
            return audio_path

        except Exception as e:
            raise TTSGenerationError(f"Mock TTS generation failed: {e}", text=text) from e

    def _generate_silent_wav(self, path: Path, duration_seconds: float) -> None:
        """Generate a silent WAV file.

        Args:
            path: Output file path.
            duration_seconds: Duration in seconds.
        """
        sample_rate = 22050
        num_frames = int(sample_rate * duration_seconds)
        silent_data = b"\x00\x00" * num_frames  # 16-bit silence

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silent_data)

    def _get_cache_key(self, text: str, language: Language) -> str:
        """Generate a cache key from text and language."""
        content = f"{text}_{language.value}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class EdgeTTSGenerator:
    """Production TTS generator using Microsoft Edge TTS (cloud-based).

    Works on both CPU and GPU machines since processing happens on Microsoft servers.
    Requires internet connection. Free but subject to rate limits.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the Edge TTS generator.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._output_dir = Path(settings.storage_path) / "tts_audio"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._voice_en = settings.tts_voice_en
        self._voice_hi = settings.tts_voice_hi

    async def generate(
        self,
        *,
        text: str,
        language: Language = Language.ENGLISH,
        output_path: Path | None = None,
    ) -> Path:
        """Generate speech audio from text using Edge TTS.

        Args:
            text: The text to convert to speech.
            language: Target language for TTS.
            output_path: Optional specific output path.

        Returns:
            Path to the generated audio file.

        Raises:
            TTSGenerationError: If generation fails.
        """
        logger.info("Generating TTS for: %s...", text[:30])

        try:
            import edge_tts

            if output_path:
                audio_path = output_path
            else:
                text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
                audio_path = self._output_dir / f"tts_{text_hash}.wav"

            # Map language to voice
            voice = self._voice_en if language == Language.ENGLISH else self._voice_hi

            # Generate audio using edge-tts
            # edge_tts.Communicate creates a generator that yields audio chunks
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(audio_path))

            logger.info("Generated TTS: %s", audio_path)
            return audio_path

        except ImportError as e:
            raise TTSGenerationError(
                "edge-tts library not available. Install with: pip install edge-tts"
            ) from e
        except Exception as e:
            raise TTSGenerationError(f"TTS generation failed: {e}", text=text) from e

    def unload(self) -> None:
        """No-op for Edge TTS (cloud-based, no local model to unload)."""
        logger.debug("Edge TTS unload called (no-op, cloud-based)")


class AI4BharatTTSGenerator:
    """Production TTS using AI4Bharat Indic TTS.

    Specialized for Hindi and other Indic languages.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the AI4Bharat TTS generator.

        Args:
            settings: Application settings.
        """
        self._output_dir = Path(settings.storage_path) / "tts_audio"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the TTS model."""
        if self._model is not None:
            return

        logger.info("Loading AI4Bharat TTS model...")

        try:
            # This would use the AI4Bharat Indic Parler TTS model
            # from transformers import AutoModelForTextToWaveform, AutoProcessor
            # self._model = ...
            raise NotImplementedError(
                "AI4Bharat integration requires specific model setup. "
                "Use EdgeTTSGenerator for now."
            )

        except Exception as e:
            raise TTSGenerationError(f"Failed to load AI4Bharat TTS: {e}") from e

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
        """
        self._load_model()
        # Implementation would go here
        raise NotImplementedError("AI4Bharat TTS not fully implemented")
