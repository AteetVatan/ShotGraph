"""Video generation service implementations."""

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.exceptions import VideoGenerationError
from core.services.device_utils import (
    cleanup_memory,
    get_hf_cache_dir,
    load_model_with_cache_check,
)

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class MockVideoGenerator:
    """Mock video generator for DEBUG_CPU mode.

    Generates placeholder videos with text overlay for testing
    the pipeline without requiring GPU resources.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the mock video generator.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._output_dir = Path(settings.storage_path) / "mock_videos"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Path] = {}
        self._resolution = tuple(map(int, settings.video_resolution.split("x")))
        self._fps = settings.video_fps

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        init_image: Path | None = None,
        seed: int | None = None,
    ) -> Path:
        """Generate a placeholder video with text overlay.

        Args:
            prompt: Visual description (displayed on video).
            duration_seconds: Duration of the clip.
            init_image: Optional initial frame (ignored in mock).
            seed: Optional random seed (used in cache key).

        Returns:
            Path to the generated video file.
        """
        # Create cache key from prompt, duration, and seed
        cache_key = self._get_cache_key(prompt, duration_seconds, seed)
        if cache_key in self._cache and self._cache[cache_key].exists():
            logger.debug("Using cached mock video: %s", cache_key)
            return self._cache[cache_key]

        logger.info("Generating mock video for: %s...", prompt[:50])

        try:
            from PIL import Image, ImageDraw

            # Create placeholder image with text
            width, height = self._resolution
            img = Image.new("RGB", (width, height), color=(48, 48, 64))
            draw = ImageDraw.Draw(img)

            # Draw mock label
            draw.text((20, 20), "[MOCK VIDEO]", fill=(255, 200, 0))

            # Draw prompt text (wrapped)
            wrapped_text = self._wrap_text(prompt, max_chars=60)
            y_offset = height // 3
            for line in wrapped_text[:6]:  # Limit to 6 lines
                draw.text((40, y_offset), line, fill=(255, 255, 255))
                y_offset += 30

            # Draw duration info
            draw.text(
                (20, height - 40),
                f"Duration: {duration_seconds}s",
                fill=(150, 150, 150),
            )

            # Save image
            img_path = self._output_dir / f"{cache_key}.png"
            img.save(img_path)

            # Convert to video using moviepy
            try:
                from moviepy import ImageClip  # type: ignore[import-untyped]
            except ImportError as e:
                raise VideoGenerationError(
                    "moviepy not installed. Install with: pip install moviepy"
                ) from e

            video_path = self._output_dir / f"{cache_key}.mp4"
            clip = ImageClip(str(img_path)).set_duration(duration_seconds)
            clip.write_videofile(
                str(video_path),
                fps=self._fps,
                codec=self._settings.video_codec,
                audio=False,
                logger=None,
            )
            clip.close()

            self._cache[cache_key] = video_path
            logger.info("Generated mock video: %s", video_path)
            return video_path

        except Exception as e:
            raise VideoGenerationError(f"Mock video generation failed: {e}") from e

    def _get_cache_key(self, prompt: str, duration: float, seed: int | None = None) -> str:
        """Generate a cache key from prompt, duration, and seed."""
        content = f"{prompt}_{duration}_{seed}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _wrap_text(self, text: str, max_chars: int = 60) -> list[str]:
        """Wrap text to fit within max characters per line."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines


class DiffuseVideoGenerator:
    """Production video generator using diffusion models.

    Uses HuggingFace diffusers library for text-to-video generation.
    Supports both GPU (CUDA) and CPU execution. GPU recommended for performance.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the diffusion video generator.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._model_path = settings.video_model_path or "stabilityai/stable-video-diffusion-img2vid"
        self._output_dir = Path(settings.storage_path) / "videos"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._resolution = tuple(map(int, settings.video_resolution.split("x")))
        self._fps = settings.video_fps
        self._pipeline = None
        self._image_pipeline = None
        self._device: str | None = None  # Will be set in _load_pipelines

    def _load_pipelines(self) -> None:
        """Lazy load the diffusion pipelines."""
        if self._pipeline is not None:
            return

        logger.info("Loading diffusion pipelines...")

        try:
            import torch
        except ImportError as e:
            raise VideoGenerationError(
                "PyTorch not installed. Install with: pip install torch"
            ) from e

        try:
            # import os
            # os.environ["XFORMERS_MORE_DETAILS"] = "1"
            from diffusers import (
                StableDiffusionPipeline,
                StableVideoDiffusionPipeline,
            )
        except ImportError as e:
            raise VideoGenerationError(
                "diffusers library not available. Install with: pip install diffusers"
            ) from e
        except (RuntimeError, OSError) as e:
            error_msg = str(e)
            if "DLL load failed" in error_msg or "_C" in error_msg or "could not be found" in error_msg:
                raise VideoGenerationError(
                    "Failed to load diffusers due to missing system dependencies. "
                    "On Windows, this usually indicates missing Visual C++ Redistributables. "
                    "Solutions:\n"
                    "1. Install Visual C++ Redistributables: "
                    "https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                    "2. Reinstall PyTorch and diffusers: "
                    "pip uninstall torch diffusers && pip install torch diffusers\n"
                    "3. Ensure PyTorch and diffusers versions are compatible\n"
                    f"Original error: {error_msg}"
                ) from e
            raise VideoGenerationError(f"Failed to import diffusers: {error_msg}") from e

        try:
            # Detect device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Using device: %s", self._device)

            # Use appropriate dtype for device
            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

            cache_dir = get_hf_cache_dir(self._settings)

            # Load image pipeline for generating keyframes with cache check
            def load_image_pipeline(name: str, **kwargs: Any) -> Any:
                return StableDiffusionPipeline.from_pretrained(
                    name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    **kwargs,
                )

            self._image_pipeline = load_model_with_cache_check(
                "sd2-community/stable-diffusion-2-1",
                load_image_pipeline,
                cache_dir=cache_dir,
                settings=self._settings,
            ).to(self._device)

            # Load video pipeline with cache check
            def load_video_pipeline(name: str, **kwargs: Any) -> Any:
                return StableVideoDiffusionPipeline.from_pretrained(
                    name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    **kwargs,
                )

            self._pipeline = load_model_with_cache_check(
                self._model_path,
                load_video_pipeline,
                cache_dir=cache_dir,
                settings=self._settings,
            ).to(self._device)

            logger.info("Diffusion pipelines loaded successfully")

        except Exception as e:
            raise VideoGenerationError(f"Failed to load diffusion pipeline: {e}") from e

    def _get_device_params(self) -> dict[str, int]:
        """Get device-specific generation parameters.

        Returns:
            Dictionary with num_frames_max, num_inference_steps, decode_chunk_size, image_steps.
        """
        if self._device == "cpu":
            return {
                "num_frames_max": 14,  # Reduced from 25
                "num_inference_steps": 20,  # Reduced from 25
                "decode_chunk_size": 1,  # Reduced from 4
                "image_steps": 20,  # For image pipeline
            }
        else:  # GPU
            return {
                "num_frames_max": 25,
                "num_inference_steps": 25,
                "decode_chunk_size": 4,
                "image_steps": 25,
            }

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        init_image: Path | None = None,
        seed: int | None = None,
    ) -> Path:
        """Generate a video clip using diffusion models.

        Args:
            prompt: Visual description for generation.
            duration_seconds: Target duration (approximate).
            init_image: Optional initial frame for continuity.
            seed: Optional random seed for reproducible generation.

        Returns:
            Path to the generated video file.
        """
        self._load_pipelines()

        logger.info("Generating video for: %s... (seed=%s)", prompt[:50], seed)

        try:
            import numpy as np
            import torch
            from PIL import Image

            # Set up generator with seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(seed)

            # Generate or load initial frame
            if init_image and init_image.exists():
                image = Image.open(init_image).convert("RGB")
                image = image.resize(self._resolution)
            else:
                # Generate keyframe from prompt
                params = self._get_device_params()
                result = self._image_pipeline(
                    prompt,
                    num_inference_steps=params["image_steps"],
                    guidance_scale=7.5,
                    generator=generator,
                )
                image = result.images[0]

            # Generate video from image
            # Note: SVD generates ~25 frames at a time
            params = self._get_device_params()
            num_frames = min(params["num_frames_max"], int(duration_seconds * self._fps / 2))

            if self._device == "cpu":
                logger.info(
                    "CPU mode: using %d frames, chunk_size=%d, steps=%d",
                    num_frames,
                    params["decode_chunk_size"],
                    params["num_inference_steps"],
                )

            # Memory cleanup before generation
            cleanup_memory(self._device)

            # Generate with retry logic for memory errors
            try:
                with torch.inference_mode():
                    frames = self._pipeline(
                        image,
                        num_frames=num_frames,
                        num_inference_steps=params["num_inference_steps"],
                        decode_chunk_size=params["decode_chunk_size"],
                        generator=generator,
                    ).frames[0]
            except RuntimeError as e:
                error_msg = str(e)
                if "not enough memory" in error_msg.lower() or "alloc" in error_msg.lower():
                    # Retry with even more conservative settings
                    logger.warning(
                        "Memory error during generation, retrying with reduced settings: %s",
                        error_msg[:100],
                    )

                    # Further reduce for retry
                    retry_params = {
                        "num_frames": min(8, num_frames),
                        "num_inference_steps": 15,
                        "decode_chunk_size": 1,
                    }

                    cleanup_memory(self._device)

                    with torch.inference_mode():
                        frames = self._pipeline(
                            image,
                            num_frames=retry_params["num_frames"],
                            num_inference_steps=retry_params["num_inference_steps"],
                            decode_chunk_size=retry_params["decode_chunk_size"],
                            generator=generator,
                        ).frames[0]
                else:
                    raise

            # Save video
            seed_suffix = f"_s{seed}" if seed else ""
            video_id = hashlib.md5(prompt.encode()).hexdigest()[:12]
            video_path = self._output_dir / f"video_{video_id}{seed_suffix}.mp4"

            # Export frames to video
            try:
                from moviepy import ImageSequenceClip  # type: ignore[import-untyped]
            except ImportError as e:
                raise VideoGenerationError(
                    "moviepy not installed. Install with: pip install moviepy"
                ) from e

            clip = ImageSequenceClip([np.array(frame) for frame in frames], fps=self._fps)
            clip.write_videofile(str(video_path), codec=self._settings.video_codec, logger=None)
            clip.close()

            logger.info("Generated video: %s", video_path)
            return video_path

        except Exception as e:
            raise VideoGenerationError(f"Video generation failed: {e}") from e

    def unload(self) -> None:
        """Unload models to free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._image_pipeline is not None:
            del self._image_pipeline
            self._image_pipeline = None

        try:
            import torch

            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Diffusion pipelines unloaded")
