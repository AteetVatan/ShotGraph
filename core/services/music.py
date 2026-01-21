"""Music generation service implementations."""

import hashlib
import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import MusicGenerationError
from core.services.device_utils import cleanup_memory

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class MockMusicGenerator:
    """Mock music generator for DEBUG_CPU mode.

    Generates silent audio files or uses a placeholder loop for testing.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the mock music generator.

        Args:
            settings: Application settings.
        """
        self._output_dir = Path(settings.storage_path) / "mock_music"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._assets_dir = Path(settings.assets_path) / "mock"
        self._cache: dict[str, Path] = {}

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        output_path: Path | None = None,
    ) -> Path:
        """Generate placeholder music (silence or looped asset).

        Args:
            prompt: Music style description (used for cache key).
            duration_seconds: Duration of the music.
            output_path: Optional specific output path.

        Returns:
            Path to the generated audio file.
        """
        cache_key = self._get_cache_key(prompt, duration_seconds)
        if cache_key in self._cache and self._cache[cache_key].exists():
            logger.debug("Using cached mock music: %s", cache_key)
            return self._cache[cache_key]

        logger.info("Generating mock music (%.1fs): %s...", duration_seconds, prompt[:30])

        try:
            if output_path:
                music_path = output_path
            else:
                music_path = self._output_dir / f"{cache_key}.wav"

            # Check for placeholder music asset
            bg_music_path = self._assets_dir / "bg_music.mp3"
            if bg_music_path.exists():
                # Loop the placeholder music to desired duration
                self._loop_audio(bg_music_path, music_path, duration_seconds)
            else:
                # Generate silent audio
                self._generate_silent_wav(music_path, duration_seconds)

            self._cache[cache_key] = music_path
            logger.info("Generated mock music: %s", music_path)
            return music_path

        except Exception as e:
            raise MusicGenerationError(f"Mock music generation failed: {e}") from e

    def _generate_silent_wav(self, path: Path, duration_seconds: float) -> None:
        """Generate a silent WAV file.

        Args:
            path: Output file path.
            duration_seconds: Duration in seconds.
        """
        sample_rate = 44100
        num_frames = int(sample_rate * duration_seconds)
        silent_data = b"\x00\x00" * num_frames * 2  # Stereo 16-bit

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silent_data)

    def _loop_audio(self, source: Path, dest: Path, duration_seconds: float) -> None:
        """Loop audio to reach desired duration.

        Args:
            source: Source audio file.
            dest: Destination file path.
            duration_seconds: Target duration.
        """
        try:
            from moviepy import AudioFileClip, concatenate_audioclips

            audio = AudioFileClip(str(source))
            loops_needed = int(duration_seconds / audio.duration) + 1
            clips = [audio] * loops_needed
            combined = concatenate_audioclips(clips).subclip(0, duration_seconds)
            combined.write_audiofile(str(dest), logger=None)
            audio.close()
            combined.close()
        except Exception:
            # Fallback to silent audio
            self._generate_silent_wav(dest, duration_seconds)

    def _get_cache_key(self, prompt: str, duration: float) -> str:
        """Generate a cache key from prompt and duration."""
        content = f"{prompt}_{duration}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class MusicGenGenerator:
    """Production music generator using Meta's MusicGen via HuggingFace Transformers.

    Generates original background music from text prompts.
    Requires GPU with ~16GB VRAM for best performance.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the MusicGen generator.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._output_dir = Path(settings.storage_path) / "music"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = settings.music_model
        self._model_org = settings.music_model_org
        self._model = None
        self._processor = None
        self._device = None

    def _load_model(self) -> None:
        """Lazy load the MusicGen model."""
        if self._model is not None:
            return

        logger.info("Loading MusicGen model: %s", self._model_name)

        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            import torch

            # Detect device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Using device: %s", self._device)

            # Normalize model name: add org prefix if not present
            if "/" not in self._model_name:
                model_name = f"{self._model_org}/{self._model_name}"
            else:
                model_name = self._model_name

            # Override model for CPU mode (memory-safe for debugging)
            # CPU mode uses musicgen-small to avoid memory exhaustion (OS error 1455 on Windows)
            if self._device == "cpu":
                original_model = model_name
                if "musicgen-medium" in model_name or "musicgen-large" in model_name:
                    model_name = f"{self._model_org}/musicgen-small"
                    logger.warning(
                        "CPU mode: downgrading model %s -> %s for memory safety",
                        original_model,
                        model_name,
                    )

            # Load model with appropriate dtype for device
            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,  # Load incrementally to reduce peak memory
            ).to(self._device)

            # Load processor
            self._processor = AutoProcessor.from_pretrained(model_name, use_safetensors=True)

            logger.info("MusicGen loaded successfully")

        except ImportError as e:
            raise MusicGenerationError(
                "transformers library not available. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise MusicGenerationError(f"Failed to load MusicGen: {e}") from e

    def _get_device_params(self) -> dict[str, float]:
        """Get device-specific generation parameters.

        Returns:
            Dictionary with max_segment_duration and tokens_per_second.
        """
        if self._device == "cpu":
            return {
                "max_segment_duration": min(6.0, self._settings.music_max_segment_duration),  # Reduced for CPU
                "tokens_per_second": 50,  # Keep same
            }
        else:  # GPU
            return {
                "max_segment_duration": self._settings.music_max_segment_duration,
                "tokens_per_second": 50,
            }

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
            duration_seconds: Duration of the music.
            output_path: Optional specific output path.

        Returns:
            Path to the generated audio file.
        """
        self._load_model()

        logger.info("Generating music (%.1fs): %s...", duration_seconds, prompt[:30])

        try:
            import scipy.io.wavfile
            import numpy as np
            import torch

            if output_path:
                music_path = output_path
            else:
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
                music_path = self._output_dir / f"music_{prompt_hash}.wav"

            # MusicGen can generate longer sequences, but we use segments for consistency
            # For longer durations, we generate segments and concatenate
            params = self._get_device_params()
            segments = []
            remaining = duration_seconds

            max_segment = params["max_segment_duration"]
            tokens_per_second = params["tokens_per_second"]

            if self._device == "cpu":
                logger.info(
                    "CPU mode: using max_segment=%.1fs, tokens_per_second=%d",
                    max_segment,
                    tokens_per_second,
                )

            while remaining > 0:
                segment_duration = min(max_segment, remaining)

                # Memory cleanup before each segment
                cleanup_memory(self._device)

                # Prepare inputs
                inputs = self._processor(
                    text=[prompt],
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)

                # Generate audio
                # max_new_tokens controls duration: ~50 tokens per second
                max_new_tokens = int(segment_duration * tokens_per_second)

                # Generate with retry logic for memory errors
                try:
                    with torch.inference_mode():
                        audio_values = self._model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                        )
                except RuntimeError as e:
                    error_msg = str(e)
                    if "not enough memory" in error_msg.lower() or "alloc" in error_msg.lower():
                        # Retry with even smaller segment
                        logger.warning(
                            "Memory error during segment generation, retrying with reduced segment: %s",
                            error_msg[:100],
                        )

                        # Further reduce segment size for retry
                        retry_segment_duration = min(3.0, segment_duration * 0.5)
                        retry_max_tokens = int(retry_segment_duration * tokens_per_second)

                        cleanup_memory(self._device)

                        with torch.inference_mode():
                            audio_values = self._model.generate(
                                **inputs,
                                max_new_tokens=retry_max_tokens,
                            )
                    else:
                        raise

                # Convert to numpy array
                # audio_values shape: [batch_size, num_channels, sequence_length]
                # Remove batch dimension and convert to numpy
                audio_array = audio_values[0].cpu().numpy()

                segments.append(audio_array)
                remaining -= segment_duration

            # Concatenate segments along the time axis (last dimension)
            if len(segments) > 1:
                # Concatenate along time axis: [channels, time1] + [channels, time2] -> [channels, time1+time2]
                full_audio = np.concatenate(segments, axis=-1)
            else:
                full_audio = segments[0]

            # Get sample rate from model config (typically 32000 for MusicGen)
            sample_rate = getattr(
                self._model.config,
                "audio_encoder",
                type("obj", (object,), {"sampling_rate": 32000})(),
            ).sampling_rate

            # Ensure audio is in the right format for scipy
            # scipy.io.wavfile.write expects shape [samples, channels] or [samples] for mono
            # HuggingFace MusicGen outputs [channels, samples]
            if full_audio.ndim == 2:
                # [channels, samples] -> [samples, channels]
                full_audio = full_audio.T
            elif full_audio.ndim == 1:
                # Mono audio, keep as is
                pass

            # Normalize to int16 range if needed
            if full_audio.dtype != np.int16:
                # Normalize to [-1, 1] range if in float format
                max_val = np.abs(full_audio).max()
                if max_val > 1.0:
                    full_audio = full_audio / max_val
                # Convert to int16
                full_audio = (full_audio * 32767).astype(np.int16)

            # Save as WAV
            scipy.io.wavfile.write(
                str(music_path),
                sample_rate,
                full_audio,
            )

            logger.info("Generated music: %s", music_path)
            return music_path

        except Exception as e:
            raise MusicGenerationError(f"Music generation failed: {e}") from e

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        try:
            import torch

            if self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("MusicGen unloaded")
