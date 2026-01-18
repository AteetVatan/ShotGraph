"""Video composition agent."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import CompositionError
from core.models import SceneList
from core.services.video_editing import NarrationSegment, SceneMusicTrack, VideoEditor

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


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
            # Collect all video clips and audio in order
            video_clips: list[Path] = []
            audio_tracks: list[Path | None] = []
            subtitles: list[tuple[float, float, str]] = []
            scene_music_tracks: list[SceneMusicTrack] = []
            narration_segments: list[NarrationSegment] = []

            current_time = 0.0

            for scene in scene_list.scenes:
                scene_start_time = current_time

                for shot in scene.shots:
                    if shot.video_file_path:
                        video_clips.append(Path(shot.video_file_path))

                        # Add audio track
                        audio_path = Path(shot.audio_file_path) if shot.audio_file_path else None
                        audio_tracks.append(audio_path)

                        # Track narration segments for audio ducking
                        if audio_path and audio_path.exists():
                            narration_segments.append(NarrationSegment(
                                start_time=current_time,
                                end_time=current_time + shot.duration_seconds,
                                audio_path=audio_path,
                            ))

                        # Add subtitle if present
                        if shot.subtitle_text or shot.dialogue:
                            text = shot.subtitle_text or shot.dialogue
                            subtitles.append((
                                current_time,
                                current_time + shot.duration_seconds,
                                text,
                            ))

                        current_time += shot.duration_seconds

                # Track music for each scene with timing
                if scene.music_file_path:
                    scene_music_tracks.append(SceneMusicTrack(
                        path=Path(scene.music_file_path),
                        start_time=scene_start_time,
                        end_time=current_time,
                        scene_id=scene.id,
                    ))

            if not video_clips:
                raise CompositionError("No video clips available for composition")

            self.logger.info(
                "Composing %d clips, %d audio tracks, %d subtitles, %d music tracks",
                len(video_clips),
                len([a for a in audio_tracks if a]),
                len(subtitles),
                len(scene_music_tracks),
            )

            # Compose the final video with per-scene music and ducking
            final_path = self._editor.compose_video(
                video_clips=video_clips,
                audio_tracks=audio_tracks,
                scene_music_tracks=scene_music_tracks if scene_music_tracks else None,
                narration_segments=narration_segments if narration_segments else None,
                subtitles=subtitles if subtitles else None,
                output_name="final_video.mp4",
                transitions=True,
                enable_ducking=True,
            )

            self.logger.info("Video composition complete: %s", final_path)
            return final_path

        except CompositionError:
            raise
        except Exception as e:
            raise CompositionError(f"Composition failed: {e}", stage="compositing") from e
