"""Video generation agent."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.models import Shot
from core.protocols.video_generator import IVideoGenerator

from .base import BaseAgent

if TYPE_CHECKING:
    from core.models import Scene
    from core.services.style_context import StyleContextManager

logger = logging.getLogger(__name__)


class VideoGenerationAgent(BaseAgent[Shot, Path]):
    """Agent that generates video clips for shots.

    Uses the injected video generator service to create video clips
    from shot descriptions. Supports continuity through last-frame
    initialization and style context for visual consistency.
    """

    def __init__(
        self,
        *,
        video_generator: IVideoGenerator,
        max_retries: int = 2,
        style_context_manager: "StyleContextManager | None" = None,
    ):
        """Initialize the video generation agent.

        Args:
            video_generator: Service for generating video clips.
            max_retries: Maximum retry attempts.
            style_context_manager: Optional style context for consistency.
        """
        super().__init__(max_retries=max_retries)
        self._generator = video_generator
        self._style_ctx = style_context_manager
        self._last_frame: Path | None = None
        self._current_scene: "Scene | None" = None
        self._shot_index: int = 0

    def set_style_context_manager(self, manager: "StyleContextManager") -> None:
        """Set the style context manager.

        Args:
            manager: The style context manager to use.
        """
        self._style_ctx = manager

    def set_current_scene(self, scene: "Scene", shot_index: int = 0) -> None:
        """Set the current scene for style context.

        Args:
            scene: The scene being processed.
            shot_index: Index of the current shot.
        """
        self._current_scene = scene
        self._shot_index = shot_index

    async def _execute(self, shot: Shot) -> Path:
        """Generate a video clip for the shot.

        Args:
            shot: The shot to generate video for.

        Returns:
            Path to the generated video file.
        """
        self.logger.info(
            "Generating video for scene %d, shot %d: %s...",
            shot.scene_id,
            shot.id,
            shot.description[:50],
        )

        # Build enhanced prompt with style context
        prompt = self._build_enhanced_prompt(shot)
        
        # Get generation hints from style context
        seed = None
        if self._style_ctx and self._current_scene:
            ctx = self._style_ctx.build_context_for_shot(
                self._current_scene,
                self._shot_index,
            )
            hints = self._style_ctx.get_video_generation_hints(ctx)
            seed = hints.get("seed")

        video_path = self._generator.generate(
            prompt=prompt,
            duration_seconds=shot.duration_seconds,
            init_image=self._last_frame,
            seed=seed,
        )

        # Cache last frame for continuity
        self._last_frame = self._extract_last_frame(video_path)
        self._shot_index += 1

        self.logger.info("Generated video: %s", video_path)
        return video_path

    def _build_enhanced_prompt(self, shot: Shot) -> str:
        """Build enhanced prompt with style context.

        Args:
            shot: The shot to build prompt for.

        Returns:
            Enhanced prompt string.
        """
        prompt = shot.description

        # Add style hints from context
        if self._style_ctx and self._current_scene:
            ctx = self._style_ctx.build_context_for_shot(
                self._current_scene,
                self._shot_index,
            )
            hints = self._style_ctx.get_video_generation_hints(ctx)
            
            style_suffix = hints.get("style_suffix", "")
            if style_suffix:
                prompt = f"{prompt}, {style_suffix}"
                self.logger.debug("Added style suffix: %s", style_suffix)

        return prompt

    def _extract_last_frame(self, video_path: Path) -> Path | None:
        """Extract the last frame from a video for continuity.

        Args:
            video_path: Path to the video file.

        Returns:
            Path to the extracted frame, or None if extraction fails.
        """
        try:
            from moviepy.editor import VideoFileClip
            from PIL import Image

            clip = VideoFileClip(str(video_path))
            # Get frame slightly before end to avoid black frames
            frame_time = max(0, clip.duration - 0.1)
            frame = clip.get_frame(frame_time)
            clip.close()

            frame_path = video_path.with_suffix(".last_frame.png")
            Image.fromarray(frame).save(frame_path)

            self.logger.debug("Extracted last frame: %s", frame_path)
            return frame_path

        except Exception as e:
            self.logger.warning("Failed to extract last frame: %s", e)
            return None

    def reset_continuity(self) -> None:
        """Reset the continuity state (clear last frame cache)."""
        self._last_frame = None
        self.logger.debug("Continuity state reset")
