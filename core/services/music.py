"""Music generation service implementations."""

import hashlib
import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.exceptions import MusicGenerationError
from core.services.device_utils import (
    cleanup_memory,
    get_hf_cache_dir,
    load_model_with_cache_check,
)

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

    def _check_local_cache_conflicts(self, model_name: str) -> bool:
        """Check for incomplete or corrupted local cache.

        Args:
            model_name: Full model name (e.g., 'facebook/musicgen-medium').

        Returns:
            True if potential cache conflict detected, False otherwise.
        """
        from pathlib import Path

        # Determine cache base directory
        if self._settings.hf_home:
            cache_base = Path(self._settings.hf_home) / "hub"
        else:
            cache_base = Path.home() / ".cache" / "huggingface" / "hub"

        # HuggingFace cache structure: {cache_base}/models--{org}--{model}
        cache_name = model_name.replace("/", "--")
        cache_path = cache_base / f"models--{cache_name}"

        if not cache_path.exists():
            return False

        # Check if cache directory exists but is incomplete
        # Look for model index file that indicates complete download
        index_file = cache_path / "refs" / "main" / "model.safetensors.index.json"
        if cache_path.exists() and not index_file.exists():
            # Check if there are any files at all
            has_files = any(cache_path.rglob("*"))
            if has_files:
                logger.warning(
                    "Potential incomplete cache detected for %s at %s. "
                    "Consider clearing cache if model loading fails.",
                    model_name,
                    cache_path,
                )
                return True

        return False

    def _load_model_with_retry(
        self,
        *,
        model_name: str,
        load_func,
        cache_dir: str | None = None,
        token: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Load HuggingFace model/processor with retry logic.

        Args:
            model_name: Full model name.
            load_func: Function to call for loading (from_pretrained).
            cache_dir: Optional cache directory path.
            token: Optional HuggingFace token for authentication.
            max_retries: Maximum retry attempts.
            retry_delay: Initial delay between retries (exponential backoff).

        Returns:
            Loaded model or processor.

        Raises:
            MusicGenerationError: If all retry attempts fail.
        """
        from pathlib import Path
        import time

        for attempt in range(max_retries):
            try:
                logger.debug("Loading %s (attempt %d/%d)", model_name, attempt + 1, max_retries)
                # Build kwargs for from_pretrained
                kwargs: dict[str, str | None] = {}
                if token:
                    kwargs["token"] = token
                if cache_dir:
                    kwargs["cache_dir"] = cache_dir
                return load_func(model_name, **kwargs)
            except OSError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Model load failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        retry_delay,
                        str(e)[:200],
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed - determine cache path for error message
                    if cache_dir:
                        cache_base = Path(cache_dir)
                    else:
                        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
                    cache_path = cache_base / f"models--{model_name.replace('/', '--')}"
                    raise MusicGenerationError(
                        f"Failed to load model '{model_name}' after {max_retries} attempts. "
                        f"Error: {e}\n\n"
                        f"Troubleshooting:\n"
                        f"1. Check network connectivity to HuggingFace\n"
                        f"2. Verify model name is correct: {model_name}\n"
                        f"3. Check for local directory conflicts: {cache_path}\n"
                        f"4. If model requires authentication, set HUGGINGFACE_TOKEN in .env\n"
                        f"5. Try manually downloading: huggingface-cli download {model_name}"
                    ) from e
            except Exception as e:
                # Non-OSError exceptions should not be retried
                raise MusicGenerationError(f"Failed to load model '{model_name}': {e}") from e

    def _load_model(self) -> None:
        """Lazy load the MusicGen model."""
        if self._model is not None and self._processor is not None:
            return

        logger.info("Loading MusicGen model: %s", self._model_name)

        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            import torch

            cache_dir = get_hf_cache_dir(self._settings)
            hf_token = self._settings.huggingface_token if self._settings.huggingface_token else None

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

            # Load model with cache check
            def load_model_func(name: str, **kwargs: Any) -> Any:
                # Only pass token if it has a value (avoid passing None)
                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "use_safetensors": True,
                    "low_cpu_mem_usage": True,
                    **kwargs,
                }
                if hf_token:
                    model_kwargs["token"] = hf_token
                return MusicgenForConditionalGeneration.from_pretrained(
                    name,
                    **model_kwargs,
                )

            self._model = load_model_with_cache_check(
                model_name=model_name,
                load_func=load_model_func,
                cache_dir=cache_dir,
                settings=self._settings,
            ).to(self._device)

            # Load processor with cache check
            def load_processor_func(name: str, **kwargs: Any) -> Any:
                # Only pass token if it has a value (avoid passing None)
                processor_kwargs = {**kwargs}
                if hf_token:
                    processor_kwargs["token"] = hf_token
                return AutoProcessor.from_pretrained(
                    name,
                    **processor_kwargs,
                )

            # DEBUG: Wrap processor loading with full traceback capture
            import traceback
            try:
                self._processor = load_model_with_cache_check(
                    model_name=model_name,
                    load_func=load_processor_func,
                    cache_dir=cache_dir,
                    settings=self._settings,
                )
            except Exception as processor_error:
                print("=" * 70)
                print("FULL TRACEBACK FOR PROCESSOR LOADING ERROR:")
                print("=" * 70)
                traceback.print_exc()
                print("=" * 70)
                print(f"Error type: {type(processor_error).__name__}")
                print(f"Error message: {processor_error}")
                print("=" * 70)
                raise

            # Validate that both model and processor loaded successfully
            if self._processor is None:
                raise MusicGenerationError("Failed to load processor: returned None")
            if self._model is None:
                raise MusicGenerationError("Failed to load model: returned None")

            logger.info("MusicGen loaded successfully")

        except ImportError as e:
            error_msg = str(e)
            # Check for specific missing dependencies
            if "protobuf" in error_msg.lower():
                raise MusicGenerationError(
                    "protobuf library is required but not installed. "
                    "Install with: pip install protobuf>=4.25.0\n"
                    f"Original error: {error_msg}"
                ) from e
            elif "transformers" in error_msg.lower():
                raise MusicGenerationError(
                    "transformers library not available. Install with: pip install transformers"
                ) from e
            else:
                raise MusicGenerationError(
                    f"Missing required dependency. {error_msg}\n"
                    "Please install missing dependencies from requirements.txt"
                ) from e
        except MusicGenerationError:
            # Re-raise our custom errors as-is
            # Reset state on partial failure to force full reload on retry
            if self._model is not None and self._processor is None:
                logger.warning("Resetting model state due to partial loading failure")
                self._model = None
            raise
        except Exception as e:
            # Reset state on partial failure to force full reload on retry
            if self._model is not None and self._processor is None:
                logger.warning("Resetting model state due to partial loading failure")
                self._model = None
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

        # Safety check: ensure processor is loaded
        if self._processor is None:
            raise MusicGenerationError(
                "Processor not loaded. Model loading may have failed partially."
            )

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
