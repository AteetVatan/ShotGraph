"""Video generation service implementations."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.protocols.video_generator import VideoGenerationResult

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
    ) -> "VideoGenerationResult":
        """Generate a placeholder video with text overlay.

        Args:
            prompt: Visual description (displayed on video).
            duration_seconds: Duration of the clip.
            init_image: Optional initial frame (ignored in mock).
            seed: Optional random seed (used in cache key).

        Returns:
            VideoGenerationResult with path and actual duration.
        """
        from core.protocols.video_generator import VideoGenerationResult
        # Create cache key from prompt, duration, and seed
        cache_key = self._get_cache_key(prompt, duration_seconds, seed)
        if cache_key in self._cache and self._cache[cache_key].exists():
            logger.debug("Using cached mock video: %s", cache_key)
            return VideoGenerationResult(
                path=self._cache[cache_key], actual_duration=duration_seconds
            )

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
            return VideoGenerationResult(path=video_path, actual_duration=duration_seconds)

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
                "num_frames_max": 14,
                "num_inference_steps": 20,
                "decode_chunk_size": 1,
                "image_steps": 20,
            }
        return {
            "num_frames_max": 25,
            "num_inference_steps": 25,
            "decode_chunk_size": 4,
            "image_steps": 25,
        }

    def _prepare_keyframe(
        self,
        prompt: str,
        init_image: Path | None,
        generator: Any,
    ) -> Any:
        """Prepare the initial keyframe for video generation.

        Args:
            prompt: Visual description for generation.
            init_image: Optional initial frame path.
            generator: Torch random generator.

        Returns:
            PIL Image ready for SVD pipeline.
        """
        from PIL import Image

        if init_image and init_image.exists():
            image = Image.open(init_image).convert("RGB")
            return image.resize(self._resolution)

        params = self._get_device_params()
        result = self._image_pipeline(
            prompt,
            num_inference_steps=params["image_steps"],
            guidance_scale=7.5,
            generator=generator,
        )
        image = result.images[0]
        return image.resize(self._resolution)

    def _calculate_frame_params(
        self,
        duration_seconds: float,
    ) -> tuple[int, float]:
        """Calculate frame count and actual duration.

        Args:
            duration_seconds: Requested duration.

        Returns:
            Tuple of (num_frames, actual_duration).
        """
        params = self._get_device_params()
        target_frames = int(duration_seconds * self._fps)
        num_frames = min(params["num_frames_max"], target_frames)
        actual_duration = num_frames / self._fps

        if num_frames < target_frames:
            logger.warning(
                "Frame limit: requested %.2fs (%d frames), generating %.2fs (%d frames)",
                duration_seconds,
                target_frames,
                actual_duration,
                num_frames,
            )

        return num_frames, actual_duration

    def _generate_svd_frames(
        self,
        image: Any,
        num_frames: int,
        generator: Any,
    ) -> tuple[list[Any], int]:
        """Generate video frames using SVD pipeline with retry logic.

        Args:
            image: Initial keyframe.
            num_frames: Number of frames to generate.
            generator: Torch random generator.

        Returns:
            Tuple of (frames list, actual frame count).
        """
        import torch

        params = self._get_device_params()

        if self._device == "cpu":
            logger.info(
                "CPU mode: using %d frames, chunk_size=%d, steps=%d",
                num_frames,
                params["decode_chunk_size"],
                params["num_inference_steps"],
            )

        cleanup_memory(self._device)

        try:
            with torch.inference_mode():
                frames = self._pipeline(
                    image,
                    num_frames=num_frames,
                    num_inference_steps=params["num_inference_steps"],
                    decode_chunk_size=params["decode_chunk_size"],
                    generator=generator,
                ).frames[0]
            return frames, num_frames
        except RuntimeError as e:
            if not self._is_memory_error(e):
                raise
            return self._retry_with_reduced_settings(image, num_frames, generator)

    def _is_memory_error(self, error: RuntimeError) -> bool:
        """Check if error is a memory-related error."""
        error_msg = str(error).lower()
        return "not enough memory" in error_msg or "alloc" in error_msg

    def _retry_with_reduced_settings(
        self,
        image: Any,
        num_frames: int,
        generator: Any,
    ) -> tuple[list[Any], int]:
        """Retry frame generation with reduced settings."""
        import torch

        logger.warning("Memory error, retrying with reduced settings")

        retry_num_frames = min(8, num_frames)
        cleanup_memory(self._device)

        with torch.inference_mode():
            frames = self._pipeline(
                image,
                num_frames=retry_num_frames,
                num_inference_steps=15,
                decode_chunk_size=1,
                generator=generator,
            ).frames[0]
        return frames, retry_num_frames

    def _save_video_from_frames(
        self,
        frames: list[Any],
        prompt: str,
        seed: int | None,
    ) -> Path:
        """Resize frames and save to video file.

        Args:
            frames: List of frames from SVD.
            prompt: Original prompt (for filename).
            seed: Optional seed (for filename).

        Returns:
            Path to saved video file.
        """
        import numpy as np
        from PIL import Image

        try:
            from moviepy import ImageSequenceClip
        except ImportError as e:
            raise VideoGenerationError(
                "moviepy not installed. Install with: pip install moviepy"
            ) from e

        resized_frames = [
            np.array(
                Image.fromarray(np.array(f)).resize(
                    self._resolution, Image.Resampling.LANCZOS
                )
            )
            for f in frames
        ]

        seed_suffix = f"_s{seed}" if seed else ""
        video_id = hashlib.md5(prompt.encode()).hexdigest()[:12]
        video_path = self._output_dir / f"video_{video_id}{seed_suffix}.mp4"

        clip = ImageSequenceClip(resized_frames, fps=self._fps)
        clip.write_videofile(str(video_path), codec=self._settings.video_codec, logger=None)
        clip.close()

        return video_path

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        init_image: Path | None = None,
        seed: int | None = None,
    ) -> "VideoGenerationResult":
        """Generate a video clip using diffusion models.

        Args:
            prompt: Visual description for generation.
            duration_seconds: Target duration (approximate).
            init_image: Optional initial frame for continuity.
            seed: Optional random seed for reproducible generation.

        Returns:
            VideoGenerationResult with path and actual duration.
        """
        from core.protocols.video_generator import VideoGenerationResult

        self._load_pipelines()
        logger.info("Generating video for: %s... (seed=%s)", prompt[:50], seed)

        try:
            import torch

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(seed)

            image = self._prepare_keyframe(prompt, init_image, generator)
            num_frames, actual_duration = self._calculate_frame_params(duration_seconds)
            frames, actual_frames = self._generate_svd_frames(image, num_frames, generator)

            # Update duration if retry reduced frame count
            if actual_frames != num_frames:
                actual_duration = actual_frames / self._fps

            video_path = self._save_video_from_frames(frames, prompt, seed)

            logger.info("Generated video: %s (actual duration: %.2fs)", video_path, actual_duration)
            return VideoGenerationResult(path=video_path, actual_duration=actual_duration)

        except VideoGenerationError:
            raise
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
