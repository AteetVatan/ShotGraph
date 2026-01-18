"""Video editing and composition service."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.exceptions import CompositionError

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class SceneMusicTrack:
    """Music track associated with a scene."""

    path: Path
    start_time: float
    end_time: float
    scene_id: int


@dataclass
class NarrationSegment:
    """A segment of narration for audio ducking."""

    start_time: float
    end_time: float
    audio_path: Path | None


class VideoEditor:
    """Video editing service using MoviePy/FFmpeg.

    Handles composition of video clips, audio tracks,
    subtitles, and transitions.
    """

    def __init__(self, settings: "Settings"):
        """Initialize the video editor.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._output_dir = Path(settings.storage_path) / "final"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fps = settings.video_fps
        self._resolution = tuple(map(int, settings.video_resolution.split("x")))
        self._video_effects = None  # Lazy loaded

    def _get_video_effects(self):
        """Lazy load video effects service.

        Returns:
            VideoEffects instance.
        """
        if self._video_effects is None:
            from core.services.video_effects import VideoEffects
            self._video_effects = VideoEffects(self._settings)
        return self._video_effects

    def compose_video(
        self,
        *,
        video_clips: list[Path],
        audio_tracks: list[Path | None] | None = None,
        music_track: Path | None = None,
        scene_music_tracks: list[SceneMusicTrack] | None = None,
        narration_segments: list[NarrationSegment] | None = None,
        subtitles: list[tuple[float, float, str]] | None = None,
        output_name: str = "final_video.mp4",
        transitions: bool = True,
        enable_ducking: bool = True,
        use_frame_interpolation: bool | None = None,
    ) -> Path:
        """Compose multiple clips into a final video.

        Args:
            video_clips: List of video clip paths in order.
            audio_tracks: Optional list of audio tracks per clip.
            music_track: Optional single background music track (legacy).
            scene_music_tracks: Optional per-scene music tracks with crossfades.
            narration_segments: Optional narration segments for audio ducking.
            subtitles: Optional list of (start_time, end_time, text) tuples.
            output_name: Name for the output file.
            transitions: Whether to add fade transitions.
            enable_ducking: Whether to duck music during narration.
            use_frame_interpolation: Use frame interpolation for transitions (default: from settings).

        Returns:
            Path to the composed video file.
        """
        # Use setting if not specified
        if use_frame_interpolation is None:
            use_frame_interpolation = self._settings.video_use_frame_interpolation
        if not video_clips:
            raise CompositionError("No video clips provided", stage="input")

        logger.info("Composing video from %d clips", len(video_clips))

        try:
            from moviepy.editor import (
                AudioFileClip,
                CompositeAudioClip,
                VideoFileClip,
                concatenate_videoclips,
                vfx,
            )

            clips = []
            total_duration = 0.0
            clip_start_times = []

            # Load and process each clip
            for i, clip_path in enumerate(video_clips):
                if not clip_path.exists():
                    logger.warning("Video clip not found: %s", clip_path)
                    continue

                video = VideoFileClip(str(clip_path))
                clip_start_times.append(total_duration)

                # Add audio track if provided
                if audio_tracks and i < len(audio_tracks) and audio_tracks[i]:
                    audio_path = audio_tracks[i]
                    if audio_path and audio_path.exists():
                        audio = AudioFileClip(str(audio_path))
                        # Trim audio to video duration
                        if audio.duration > video.duration:
                            audio = audio.subclip(0, video.duration)
                        video = video.set_audio(audio)

                clips.append(video)
                total_duration += video.duration

            if not clips:
                raise CompositionError("No valid video clips loaded", stage="loading")

            # Apply transitions if enabled
            if transitions and len(clips) > 1:
                fade_duration = self._settings.video_transition_duration
                
                if use_frame_interpolation:
                    # Use frame interpolation for smoother transitions
                    clips = self._apply_interpolated_transitions(
                        clips,
                        video_clips,
                        fade_duration,
                    )
                else:
                    # Use standard fade transitions
                    processed_clips = []
                    for i, clip in enumerate(clips):
                        if i > 0:
                            clip = clip.fx(vfx.fadein, fade_duration)
                        if i < len(clips) - 1:
                            clip = clip.fx(vfx.fadeout, fade_duration)
                        processed_clips.append(clip)
                    clips = processed_clips

            # Concatenate all clips
            final_video = concatenate_videoclips(clips, method="compose")

            # Prepare background music
            music_audio = None

            # Use per-scene music with crossfades if provided
            if scene_music_tracks and len(scene_music_tracks) > 0:
                logger.info("Mixing %d scene music tracks", len(scene_music_tracks))
                music_audio = self._mix_scene_music(
                    scene_music_tracks,
                    total_duration,
                    crossfade_duration=1.0,
                )
            # Otherwise use single music track (legacy mode)
            elif music_track and music_track.exists():
                music = AudioFileClip(str(music_track))
                # Loop music if shorter than video
                if music.duration < final_video.duration:
                    music = music.fx(vfx.loop, duration=final_video.duration)
                else:
                    music = music.subclip(0, final_video.duration)
                music_audio = music

            # Apply audio ducking if enabled and we have narration
            if music_audio and enable_ducking and narration_segments:
                logger.info("Applying audio ducking for %d narration segments", len(narration_segments))
                music_audio = self._apply_audio_ducking(
                    music_audio,
                    narration_segments,
                    total_duration,
                )

            # Apply base music volume
            if music_audio:
                music_audio = music_audio.volumex(self._settings.music_volume)

                # Mix with existing audio
                if final_video.audio:
                    mixed_audio = CompositeAudioClip([final_video.audio, music_audio])
                    final_video = final_video.set_audio(mixed_audio)
                else:
                    final_video = final_video.set_audio(music_audio)

            # Add subtitles if provided
            if subtitles:
                final_video = self._add_subtitles(final_video, subtitles)

            # Write final video
            output_path = self._output_dir / output_name
            final_video.write_videofile(
                str(output_path),
                fps=self._fps,
                codec=self._settings.video_codec,
                audio_codec=self._settings.video_audio_codec,
                logger=None,
            )

            # Cleanup
            for clip in clips:
                clip.close()
            final_video.close()
            if music_audio:
                music_audio.close()

            logger.info("Composed video saved to: %s", output_path)
            return output_path

        except Exception as e:
            raise CompositionError(f"Video composition failed: {e}", stage="composition") from e

    def _mix_scene_music(
        self,
        scene_tracks: list[SceneMusicTrack],
        total_duration: float,
        crossfade_duration: float = 1.0,
    ):
        """Mix multiple scene music tracks with crossfades.

        Args:
            scene_tracks: List of scene music tracks with timing info.
            total_duration: Total video duration.
            crossfade_duration: Duration of crossfade between tracks.

        Returns:
            Combined audio clip with crossfaded music.
        """
        from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_audioclips

        if not scene_tracks:
            return None

        # Sort by start time
        scene_tracks = sorted(scene_tracks, key=lambda x: x.start_time)

        audio_clips = []

        for i, track in enumerate(scene_tracks):
            if not track.path.exists():
                logger.warning("Scene music not found: %s", track.path)
                continue

            try:
                audio = AudioFileClip(str(track.path))
                scene_duration = track.end_time - track.start_time

                # Loop or trim to fit scene duration
                if audio.duration < scene_duration:
                    # Loop the music
                    from moviepy.editor import vfx
                    audio = audio.fx(vfx.loop, duration=scene_duration)
                else:
                    audio = audio.subclip(0, scene_duration)

                # Apply fade in/out for crossfade effect
                if i > 0:
                    # Fade in at start (except first track)
                    audio = audio.audio_fadein(crossfade_duration)
                if i < len(scene_tracks) - 1:
                    # Fade out at end (except last track)
                    audio = audio.audio_fadeout(crossfade_duration)

                # Set start time
                audio = audio.set_start(track.start_time)
                audio_clips.append(audio)

            except Exception as e:
                logger.warning("Failed to load scene music %s: %s", track.path, e)
                continue

        if not audio_clips:
            return None

        # Composite all audio tracks (they may overlap during crossfades)
        return CompositeAudioClip(audio_clips)

    def _apply_audio_ducking(
        self,
        music,
        narration_segments: list[NarrationSegment],
        total_duration: float,
        duck_level: float = 0.2,
        fade_duration: float = 0.3,
    ):
        """Apply audio ducking to lower music volume during narration.

        Args:
            music: The music audio clip.
            narration_segments: List of narration segments with timing.
            total_duration: Total duration of the video.
            duck_level: Volume level during narration (0.0-1.0).
            fade_duration: Duration of volume fade in/out.

        Returns:
            Music clip with ducking applied.
        """
        from moviepy.audio.AudioClip import AudioClip

        # Create a volume envelope based on narration segments
        def make_volume_envelope(t):
            """Generate volume multiplier for time t."""
            if isinstance(t, np.ndarray):
                # Vectorized version for arrays
                result = np.ones_like(t)
                for seg in narration_segments:
                    # Check if t is within narration segment
                    in_segment = (t >= seg.start_time) & (t <= seg.end_time)
                    
                    # Check if t is in fade-in region (before segment)
                    fade_in_mask = (t >= seg.start_time - fade_duration) & (t < seg.start_time)
                    fade_in_progress = (t - (seg.start_time - fade_duration)) / fade_duration
                    fade_in_volume = 1.0 - (1.0 - duck_level) * fade_in_progress
                    
                    # Check if t is in fade-out region (after segment)
                    fade_out_mask = (t > seg.end_time) & (t <= seg.end_time + fade_duration)
                    fade_out_progress = (t - seg.end_time) / fade_duration
                    fade_out_volume = duck_level + (1.0 - duck_level) * fade_out_progress
                    
                    # Apply volumes
                    result = np.where(in_segment, duck_level, result)
                    result = np.where(fade_in_mask, np.minimum(result, fade_in_volume), result)
                    result = np.where(fade_out_mask, np.minimum(result, fade_out_volume), result)
                
                return result
            else:
                # Scalar version
                volume = 1.0
                for seg in narration_segments:
                    if seg.start_time <= t <= seg.end_time:
                        return duck_level
                    elif seg.start_time - fade_duration <= t < seg.start_time:
                        # Fading into duck
                        progress = (t - (seg.start_time - fade_duration)) / fade_duration
                        fade_volume = 1.0 - (1.0 - duck_level) * progress
                        volume = min(volume, fade_volume)
                    elif seg.end_time < t <= seg.end_time + fade_duration:
                        # Fading out of duck
                        progress = (t - seg.end_time) / fade_duration
                        fade_volume = duck_level + (1.0 - duck_level) * progress
                        volume = min(volume, fade_volume)
                return volume

        # Apply volume envelope to music
        def make_frame(gf, t):
            """Apply volume envelope to audio frame."""
            volume = make_volume_envelope(t)
            if isinstance(volume, np.ndarray):
                volume = volume.reshape(-1, 1)  # For stereo
            return gf(t) * volume

        # Use fl method to apply the volume modification
        ducked_music = music.fl(make_frame, keep_duration=True)
        
        logger.info("Applied ducking to %d segments (duck_level=%.2f)", len(narration_segments), duck_level)
        return ducked_music

    def _apply_interpolated_transitions(
        self,
        clips: list,
        clip_paths: list[Path],
        transition_duration: float,
    ) -> list:
        """Apply frame interpolation transitions between clips.

        Args:
            clips: List of loaded VideoFileClip objects.
            clip_paths: List of original clip paths (for frame extraction).
            transition_duration: Duration of transition.

        Returns:
            List of clips with interpolated transitions inserted.
        """
        from moviepy.editor import ImageSequenceClip, concatenate_videoclips, vfx

        effects = self._get_video_effects()
        num_interp_frames = self._settings.video_interpolation_frames
        
        result_clips = []
        
        for i, clip in enumerate(clips):
            # Add fade transitions as well
            if i > 0:
                clip = clip.fx(vfx.fadein, transition_duration / 2)
            if i < len(clips) - 1:
                clip = clip.fx(vfx.fadeout, transition_duration / 2)
            
            result_clips.append(clip)
            
            # Insert interpolated transition between clips
            if i < len(clips) - 1:
                try:
                    # Get last frame of current clip
                    last_frame = clip.get_frame(clip.duration - 0.1)
                    # Get first frame of next clip
                    first_frame = clips[i + 1].get_frame(0.1)
                    
                    # Generate interpolated frames
                    interp_frames = effects.interpolate_frames(
                        last_frame,
                        first_frame,
                        num_frames=num_interp_frames,
                    )
                    
                    if interp_frames:
                        # Create clip from interpolated frames
                        interp_clip = ImageSequenceClip(
                            interp_frames,
                            fps=self._fps,
                        )
                        result_clips.append(interp_clip)
                        logger.debug("Added interpolation transition between clips %d and %d", i, i + 1)
                except Exception as e:
                    logger.warning("Failed to create interpolated transition: %s", e)
        
        return result_clips

    def _add_subtitles(
        self,
        video,
        subtitles: list[tuple[float, float, str]],
    ):
        """Add subtitles to a video clip.

        Args:
            video: The video clip to add subtitles to.
            subtitles: List of (start_time, end_time, text) tuples.

        Returns:
            Video clip with subtitles.
        """
        from moviepy.editor import CompositeVideoClip, TextClip

        subtitle_clips = []

        for start_time, end_time, text in subtitles:
            try:
                txt_clip = (
                    TextClip(
                        text,
                        fontsize=self._settings.video_subtitle_fontsize,
                        color="white",
                        stroke_color="black",
                        stroke_width=2,
                        method="caption",
                        size=(video.w - 100, None),
                    )
                    .set_position(("center", "bottom"))
                    .set_start(start_time)
                    .set_duration(end_time - start_time)
                )
                subtitle_clips.append(txt_clip)
            except Exception as e:
                logger.warning("Failed to create subtitle: %s", e)
                continue

        if subtitle_clips:
            return CompositeVideoClip([video] + subtitle_clips)
        return video

    def extract_frame(self, video_path: Path, time: float, output_path: Path) -> Path:
        """Extract a frame from a video at a specific time.

        Args:
            video_path: Path to the video file.
            time: Time in seconds to extract frame.
            output_path: Path to save the extracted frame.

        Returns:
            Path to the extracted frame image.
        """
        try:
            from moviepy.editor import VideoFileClip
            from PIL import Image

            clip = VideoFileClip(str(video_path))
            frame = clip.get_frame(min(time, clip.duration - 0.1))
            clip.close()

            Image.fromarray(frame).save(output_path)
            return output_path

        except Exception as e:
            raise CompositionError(f"Frame extraction failed: {e}", stage="extraction") from e

    def get_video_duration(self, video_path: Path) -> float:
        """Get the duration of a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds.
        """
        try:
            from moviepy.editor import VideoFileClip

            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            clip.close()
            return duration

        except Exception as e:
            logger.warning("Failed to get video duration: %s", e)
            return 0.0
