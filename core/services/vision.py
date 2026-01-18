"""Video generation service implementations."""

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import VideoGenerationError

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
            from moviepy.editor import ImageClip

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
    Requires GPU with sufficient VRAM.
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

    def _load_pipelines(self) -> None:
        """Lazy load the diffusion pipelines."""
        if self._pipeline is not None:
            return

        logger.info("Loading diffusion pipelines...")

        try:
            import torch
            from diffusers import (
                StableDiffusionPipeline,
                StableVideoDiffusionPipeline,
            )

            # Load image pipeline for generating keyframes
            self._image_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16,
            ).to("cuda")

            # Load video pipeline
            self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16,
            ).to("cuda")

            logger.info("Diffusion pipelines loaded successfully")

        except ImportError as e:
            raise VideoGenerationError(
                "diffusers library not available. Install with: pip install diffusers"
            ) from e
        except Exception as e:
            raise VideoGenerationError(f"Failed to load diffusion pipeline: {e}") from e

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
            import torch
            from PIL import Image

            # Set up generator with seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed)

            # Generate or load initial frame
            if init_image and init_image.exists():
                image = Image.open(init_image).convert("RGB")
                image = image.resize(self._resolution)
            else:
                # Generate keyframe from prompt
                result = self._image_pipeline(
                    prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=generator,
                )
                image = result.images[0]

            # Generate video from image
            # Note: SVD generates ~25 frames at a time
            num_frames = min(25, int(duration_seconds * self._fps / 2))

            with torch.inference_mode():
                frames = self._pipeline(
                    image,
                    num_frames=num_frames,
                    num_inference_steps=25,
                    decode_chunk_size=4,
                    generator=generator,
                ).frames[0]

            # Save video
            seed_suffix = f"_s{seed}" if seed else ""
            video_id = hashlib.md5(prompt.encode()).hexdigest()[:12]
            video_path = self._output_dir / f"video_{video_id}{seed_suffix}.mp4"

            # Export frames to video
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip([frame for frame in frames], fps=self._fps)
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

            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Diffusion pipelines unloaded")
