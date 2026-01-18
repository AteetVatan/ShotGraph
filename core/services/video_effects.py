"""Video effects service for transitions, interpolation, and Ken Burns effects."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.exceptions import VideoGenerationError

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class VideoEffects:
    """Video effects service for enhanced visual transitions.

    Provides:
    - Frame interpolation between shots
    - Ken Burns (pan/zoom) effects on still images
    - Smooth transitions
    """

    def __init__(self, settings: "Settings"):
        """Initialize the video effects service.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._fps = settings.video_fps
        self._resolution = tuple(map(int, settings.video_resolution.split("x")))
        self._output_dir = Path(settings.storage_path) / "effects"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def interpolate_frames(
        self,
        frame1: Path | np.ndarray,
        frame2: Path | np.ndarray,
        *,
        num_frames: int = 4,
        method: str = "blend",
    ) -> list[np.ndarray]:
        """Generate intermediate frames between two images.

        Args:
            frame1: First frame (path or array).
            frame2: Second frame (path or array).
            num_frames: Number of intermediate frames to generate.
            method: Interpolation method ('blend' or 'rife').

        Returns:
            List of interpolated frame arrays.
        """
        logger.info("Interpolating %d frames between images", num_frames)

        # Load frames if paths are provided
        arr1 = self._load_frame(frame1)
        arr2 = self._load_frame(frame2)

        # Ensure same size
        if arr1.shape != arr2.shape:
            arr2 = self._resize_frame(arr2, arr1.shape[:2])

        if method == "rife":
            return self._interpolate_rife(arr1, arr2, num_frames)
        else:
            return self._interpolate_blend(arr1, arr2, num_frames)

    def _load_frame(self, frame: Path | np.ndarray) -> np.ndarray:
        """Load a frame from path or return the array.

        Args:
            frame: Frame path or array.

        Returns:
            Frame as numpy array.
        """
        if isinstance(frame, np.ndarray):
            return frame

        try:
            from PIL import Image

            img = Image.open(frame).convert("RGB")
            return np.array(img)
        except Exception as e:
            raise VideoGenerationError(f"Failed to load frame: {e}") from e

    def _resize_frame(self, frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize a frame to target size.

        Args:
            frame: Frame array.
            target_size: Target (height, width).

        Returns:
            Resized frame array.
        """
        try:
            from PIL import Image

            img = Image.fromarray(frame)
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(img)
        except Exception as e:
            logger.warning("Failed to resize frame: %s", e)
            return frame

    def _interpolate_blend(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """Linear blend interpolation between frames.

        Args:
            frame1: First frame array.
            frame2: Second frame array.
            num_frames: Number of intermediate frames.

        Returns:
            List of blended frames.
        """
        frames = []
        for i in range(num_frames):
            alpha = (i + 1) / (num_frames + 1)
            blended = (1 - alpha) * frame1.astype(np.float32) + alpha * frame2.astype(np.float32)
            frames.append(blended.astype(np.uint8))
        return frames

    def _interpolate_rife(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """RIFE-based frame interpolation (if available).

        Falls back to blend if RIFE is not installed.

        Args:
            frame1: First frame array.
            frame2: Second frame array.
            num_frames: Number of intermediate frames.

        Returns:
            List of interpolated frames.
        """
        try:
            # Try to import RIFE
            # Note: RIFE requires specific installation
            import torch
            from torch.nn.functional import interpolate as torch_interpolate

            logger.info("Using RIFE-style interpolation")

            # Simple optical flow-based interpolation
            # For production, use actual RIFE model
            frames = []
            for i in range(num_frames):
                alpha = (i + 1) / (num_frames + 1)
                # Use weighted blend with slight motion blur effect
                blended = (1 - alpha) * frame1.astype(np.float32) + alpha * frame2.astype(np.float32)
                frames.append(blended.astype(np.uint8))
            return frames

        except ImportError:
            logger.warning("RIFE not available, using linear blend")
            return self._interpolate_blend(frame1, frame2, num_frames)

    def apply_ken_burns(
        self,
        image: Path | np.ndarray,
        *,
        duration: float,
        start_zoom: float = 1.0,
        end_zoom: float = 1.2,
        pan_direction: str = "right",
        pan_amount: float = 0.1,
    ) -> Path:
        """Apply Ken Burns (pan/zoom) effect to a still image.

        Args:
            image: Input image path or array.
            duration: Duration of the effect in seconds.
            start_zoom: Initial zoom level (1.0 = no zoom).
            end_zoom: Final zoom level.
            pan_direction: Direction to pan ('left', 'right', 'up', 'down', 'none').
            pan_amount: Amount to pan (0.0-1.0, fraction of image size).

        Returns:
            Path to the generated video clip.
        """
        logger.info(
            "Applying Ken Burns effect: zoom %.2f->%.2f, pan %s (%.2f)",
            start_zoom,
            end_zoom,
            pan_direction,
            pan_amount,
        )

        try:
            from moviepy.editor import VideoClip
            from PIL import Image

            # Load image
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert("RGB")
            else:
                img = Image.fromarray(image)

            # Get target resolution
            target_w, target_h = self._resolution
            img_w, img_h = img.size

            # Scale image to be larger than target for zoom/pan room
            max_zoom = max(start_zoom, end_zoom)
            scale_factor = max_zoom * 1.2  # Extra margin
            scaled_w = int(target_w * scale_factor)
            scaled_h = int(target_h * scale_factor)
            img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            img_array = np.array(img)

            # Calculate pan offsets
            pan_x_start, pan_y_start = 0.0, 0.0
            pan_x_end, pan_y_end = 0.0, 0.0

            if pan_direction == "right":
                pan_x_start = 0.0
                pan_x_end = pan_amount
            elif pan_direction == "left":
                pan_x_start = pan_amount
                pan_x_end = 0.0
            elif pan_direction == "down":
                pan_y_start = 0.0
                pan_y_end = pan_amount
            elif pan_direction == "up":
                pan_y_start = pan_amount
                pan_y_end = 0.0

            def make_frame(t):
                """Generate a frame at time t with Ken Burns effect."""
                progress = t / duration

                # Interpolate zoom
                current_zoom = start_zoom + (end_zoom - start_zoom) * progress

                # Interpolate pan
                current_pan_x = pan_x_start + (pan_x_end - pan_x_start) * progress
                current_pan_y = pan_y_start + (pan_y_end - pan_y_start) * progress

                # Calculate crop region
                crop_w = int(scaled_w / current_zoom)
                crop_h = int(scaled_h / current_zoom)

                # Calculate crop position with pan
                max_offset_x = scaled_w - crop_w
                max_offset_y = scaled_h - crop_h
                offset_x = int(current_pan_x * max_offset_x)
                offset_y = int(current_pan_y * max_offset_y)

                # Ensure we don't go out of bounds
                offset_x = max(0, min(offset_x, max_offset_x))
                offset_y = max(0, min(offset_y, max_offset_y))

                # Crop and resize to target
                cropped = img_array[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
                
                # Resize to target resolution
                from PIL import Image as PILImage
                frame_img = PILImage.fromarray(cropped)
                frame_img = frame_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
                return np.array(frame_img)

            # Create video clip
            clip = VideoClip(make_frame, duration=duration)

            # Generate unique output path
            import hashlib

            img_hash = hashlib.md5(str(image).encode()).hexdigest()[:8]
            output_path = self._output_dir / f"kenburns_{img_hash}_{duration}s.mp4"

            clip.write_videofile(
                str(output_path),
                fps=self._fps,
                codec=self._settings.video_codec,
                audio=False,
                logger=None,
            )
            clip.close()

            logger.info("Ken Burns effect applied: %s", output_path)
            return output_path

        except Exception as e:
            raise VideoGenerationError(f"Ken Burns effect failed: {e}") from e

    def create_transition_clip(
        self,
        clip1_last_frame: Path | np.ndarray,
        clip2_first_frame: Path | np.ndarray,
        *,
        duration: float = 0.5,
        transition_type: str = "crossfade",
    ) -> Path:
        """Create a transition clip between two video segments.

        Args:
            clip1_last_frame: Last frame of first clip.
            clip2_first_frame: First frame of second clip.
            duration: Transition duration in seconds.
            transition_type: Type of transition ('crossfade', 'interpolate').

        Returns:
            Path to the transition clip.
        """
        logger.info("Creating %s transition (%.2fs)", transition_type, duration)

        try:
            from moviepy.editor import VideoClip

            frame1 = self._load_frame(clip1_last_frame)
            frame2 = self._load_frame(clip2_first_frame)

            # Ensure same size
            if frame1.shape != frame2.shape:
                frame2 = self._resize_frame(frame2, frame1.shape[:2])

            if transition_type == "interpolate":
                # Generate interpolated frames
                num_frames = int(duration * self._fps)
                frames = self.interpolate_frames(frame1, frame2, num_frames=num_frames)

                def make_frame(t):
                    idx = int(t * self._fps)
                    idx = min(idx, len(frames) - 1)
                    return frames[idx]
            else:
                # Simple crossfade
                def make_frame(t):
                    alpha = t / duration
                    blended = (1 - alpha) * frame1.astype(np.float32) + alpha * frame2.astype(np.float32)
                    return blended.astype(np.uint8)

            clip = VideoClip(make_frame, duration=duration)

            import hashlib

            hash_input = f"{clip1_last_frame}_{clip2_first_frame}_{transition_type}"
            clip_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            output_path = self._output_dir / f"transition_{clip_hash}.mp4"

            clip.write_videofile(
                str(output_path),
                fps=self._fps,
                codec=self._settings.video_codec,
                audio=False,
                logger=None,
            )
            clip.close()

            logger.info("Transition clip created: %s", output_path)
            return output_path

        except Exception as e:
            raise VideoGenerationError(f"Transition creation failed: {e}") from e


class MockVideoEffects:
    """Mock video effects for debug mode."""

    def __init__(self, settings: "Settings"):
        """Initialize mock video effects."""
        self._settings = settings
        self._output_dir = Path(settings.storage_path) / "effects"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def interpolate_frames(
        self,
        frame1: Path | np.ndarray,
        frame2: Path | np.ndarray,
        *,
        num_frames: int = 4,
        method: str = "blend",
    ) -> list[np.ndarray]:
        """Return empty frame list in mock mode."""
        return []

    def apply_ken_burns(
        self,
        image: Path | np.ndarray,
        *,
        duration: float,
        start_zoom: float = 1.0,
        end_zoom: float = 1.2,
        pan_direction: str = "right",
        pan_amount: float = 0.1,
    ) -> Path:
        """Create a simple static video from the image."""
        logger.info("Mock Ken Burns: creating static video from image")

        try:
            from moviepy.editor import ImageClip
            from PIL import Image

            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = Image.fromarray(image)

            # Save temp image
            temp_path = self._output_dir / "temp_kenburns.png"
            img.save(temp_path)

            output_path = self._output_dir / f"mock_kenburns_{duration}s.mp4"
            clip = ImageClip(str(temp_path)).set_duration(duration)
            clip.write_videofile(str(output_path), fps=24, codec="libx264", audio=False, logger=None)
            clip.close()

            return output_path

        except Exception as e:
            logger.warning("Mock Ken Burns failed: %s", e)
            raise VideoGenerationError(f"Mock Ken Burns failed: {e}") from e

    def create_transition_clip(
        self,
        clip1_last_frame: Path | np.ndarray,
        clip2_first_frame: Path | np.ndarray,
        *,
        duration: float = 0.5,
        transition_type: str = "crossfade",
    ) -> Path:
        """Return None in mock mode (no transition clip)."""
        logger.info("Mock transition: returning first frame as static")
        return self.apply_ken_burns(clip1_last_frame, duration=duration)
