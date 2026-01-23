"""Video composition agent."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from moviepy import VideoFileClip

from core.exceptions import CompositionError
from core.models import Scene, SceneList, Shot
from core.services.video_editing import NarrationSegment, SceneMusicTrack, VideoEditor

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class CompositionAssets:
    """Collected assets for video composition."""

    video_clips: list[Path] = field(default_factory=list)
    audio_tracks: list[Path | None] = field(default_factory=list)
    subtitles: list[tuple[float, float, str]] = field(default_factory=list)
    scene_music_tracks: list[SceneMusicTrack] = field(default_factory=list)
    narration_segments: list[NarrationSegment] = field(default_factory=list)


class VideoCompositorAgent:
    """Agent that composes the final video from all assets.

    Combines video clips, audio tracks, and subtitles into a
    cohesive final video with transitions.
    """

    def __init__(self, *, settings: "Settings"):
        """Initialize the compositor agent.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._editor = VideoEditor(settings)
        self._output_dir = Path(settings.storage_path) / "final"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run(self, scene_list: SceneList) -> Path:
        """Compose the final video from all scenes.

        Args:
            scene_list: The complete scene list with generated assets.

        Returns:
            Path to the final composed video.

        Raises:
            CompositionError: If composition fails.
        """
        self.logger.info("Starting video composition for %d scenes", len(scene_list.scenes))

        try:
            assets = self._collect_assets(scene_list)

            if not assets.video_clips:
                raise CompositionError("No video clips available for composition")

            self._log_composition_summary(assets)

            final_path = self._editor.compose_video(
                video_clips=assets.video_clips,
                audio_tracks=assets.audio_tracks,
                scene_music_tracks=assets.scene_music_tracks or None,
                narration_segments=assets.narration_segments or None,
                subtitles=assets.subtitles or None,
                output_name="final_video.mp4",
                transitions=True,
                enable_ducking=True,
            )

            self.logger.info("Video composition complete: %s", final_path)
            return final_path

        except CompositionError:
            raise
        except (OSError, IOError) as e:
            raise CompositionError(f"File I/O error: {e}", stage="compositing") from e
        except Exception as e:
            raise CompositionError(f"Composition failed: {e}", stage="compositing") from e

    def _collect_assets(self, scene_list: SceneList) -> CompositionAssets:
        """Collect all assets from scenes for composition.

        Args:
            scene_list: The complete scene list.

        Returns:
            CompositionAssets with all collected data.
        """
        assets = CompositionAssets()
        current_time = 0.0

        for scene in scene_list.scenes:
            scene_start_time = current_time
            current_time = self._collect_scene_assets(scene, assets, current_time)
            self._add_scene_music(scene, assets, scene_start_time, current_time)

        return assets

    def _collect_scene_assets(
        self,
        scene: Scene,
        assets: CompositionAssets,
        current_time: float,
    ) -> float:
        """Collect assets from a single scene.

        Args:
            scene: The scene to process.
            assets: Assets container to populate.
            current_time: Current timeline position.

        Returns:
            Updated timeline position.
        """
        for shot in scene.shots:
            if shot.video_file_path:
                current_time = self._process_shot(shot, assets, current_time)
        return current_time

    def _process_shot(
        self,
        shot: Shot,
        assets: CompositionAssets,
        current_time: float,
    ) -> float:
        """Process a single shot and add its assets.

        Args:
            shot: The shot to process.
            assets: Assets container to populate.
            current_time: Current timeline position.

        Returns:
            Updated timeline position.
        """
        clip_path = Path(shot.video_file_path)  # type: ignore[arg-type]
        assets.video_clips.append(clip_path)

        actual_duration = self._get_clip_duration(clip_path, shot.duration_seconds)

        audio_path = Path(shot.audio_file_path) if shot.audio_file_path else None
        assets.audio_tracks.append(audio_path)

        if audio_path and audio_path.exists():
            assets.narration_segments.append(NarrationSegment(
                start_time=current_time,
                end_time=current_time + actual_duration,
                audio_path=audio_path,
            ))

        if shot.subtitle_text or shot.dialogue:
            text = shot.subtitle_text or shot.dialogue or ""
            assets.subtitles.append((current_time, current_time + actual_duration, text))

        return current_time + actual_duration

    def _get_clip_duration(self, clip_path: Path, fallback: float) -> float:
        """Get actual duration from video file.

        Args:
            clip_path: Path to video file.
            fallback: Fallback duration if read fails.

        Returns:
            Actual video duration in seconds.
        """
        try:
            with VideoFileClip(str(clip_path)) as clip:
                return clip.duration
        except (OSError, IOError) as e:
            self.logger.warning(
                "Failed to read video duration for %s, using fallback: %s",
                clip_path,
                e,
            )
            return fallback

    def _add_scene_music(
        self,
        scene: Scene,
        assets: CompositionAssets,
        start_time: float,
        end_time: float,
    ) -> None:
        """Add scene music track if present.

        Args:
            scene: The scene with optional music.
            assets: Assets container to populate.
            start_time: Scene start time.
            end_time: Scene end time.
        """
        if scene.music_file_path:
            assets.scene_music_tracks.append(SceneMusicTrack(
                path=Path(scene.music_file_path),
                start_time=start_time,
                end_time=end_time,
                scene_id=scene.id,
            ))

    def _log_composition_summary(self, assets: CompositionAssets) -> None:
        """Log summary of assets to be composed."""
        self.logger.info(
            "Composing %d clips, %d audio tracks, %d subtitles, %d music tracks",
            len(assets.video_clips),
            len([a for a in assets.audio_tracks if a]),
            len(assets.subtitles),
            len(assets.scene_music_tracks),
        )
