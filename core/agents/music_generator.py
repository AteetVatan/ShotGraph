"""Music generation agent."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.models import Scene
from core.protocols.music_generator import IMusicGenerator

from .base import BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MusicAgent(BaseAgent[Scene, Path]):
    """Agent that generates background music for scenes.

    Uses the injected music generator service to create background
    music that matches the mood of each scene.
    """

    def __init__(
        self,
        *,
        music_generator: IMusicGenerator,
        max_retries: int = 2,
        default_duration: float = 30.0,
    ):
        """Initialize the music agent.

        Args:
            music_generator: Service for generating music.
            max_retries: Maximum retry attempts.
            default_duration: Default music duration if no shots.
        """
        super().__init__(max_retries=max_retries)
        self._generator = music_generator
        self._default_duration = default_duration

    async def _execute(self, scene: Scene) -> Path:
        """Generate background music for the scene.

        Args:
            scene: The scene to generate music for.

        Returns:
            Path to the generated music file.
        """
        # Calculate scene duration from shots
        duration = self._calculate_scene_duration(scene)

        # Generate mood description from scene summary
        mood_prompt = self._generate_mood_prompt(scene)

        self.logger.info(
            "Generating music for scene %d (%.1fs): %s",
            scene.id,
            duration,
            mood_prompt[:50],
        )

        music_path = self._generator.generate(
            prompt=mood_prompt,
            duration_seconds=duration,
        )

        self.logger.info("Generated music: %s", music_path)
        return music_path

    def _calculate_scene_duration(self, scene: Scene) -> float:
        """Calculate total duration of a scene from its shots.

        Args:
            scene: The scene to calculate duration for.

        Returns:
            Total duration in seconds.
        """
        if not scene.shots:
            return self._default_duration

        return sum(shot.duration_seconds for shot in scene.shots)

    def _generate_mood_prompt(self, scene: Scene) -> str:
        """Generate a music mood prompt from scene description.

        Args:
            scene: The scene to analyze.

        Returns:
            A prompt describing the desired music mood.
        """
        # Basic mood extraction from scene summary
        summary_lower = scene.summary.lower()

        # Map keywords to music moods
        mood_mappings = {
            ("battle", "fight", "war", "attack"): "epic orchestral battle music, dramatic",
            ("love", "romance", "heart"): "romantic gentle music, emotional strings",
            ("sad", "death", "loss", "cry"): "melancholic music, soft piano, emotional",
            ("happy", "joy", "celebration"): "uplifting cheerful music, bright melody",
            ("fear", "scary", "horror", "dark"): "tense suspenseful music, ominous",
            ("adventure", "journey", "quest"): "adventurous orchestral music, heroic",
            ("peaceful", "calm", "nature"): "peaceful ambient music, soft nature sounds",
            ("mystery", "secret", "hidden"): "mysterious music, subtle tension",
        }

        for keywords, mood in mood_mappings.items():
            if any(kw in summary_lower for kw in keywords):
                return f"{mood}, cinematic background music"

        # Default mood
        return "cinematic background music, orchestral, atmospheric"
