"""Video editing and composition service."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    def _apply_volume(self, audio_clip: Any, volume: float) -> Any:
        """Apply volume adjustment to audio clip.

        Supports both MoviePy v1.x (volumex) and v2.x (MultiplyVolume effect).

        Args:
            audio_clip: Audio clip to adjust volume.
            volume: Volume multiplier (0.0 to 1.0+).

        Returns:
            Audio clip with volume applied.
        """
        try:
            # MoviePy v2.x: use with_effects with MultiplyVolume
            from moviepy.audio.fx import MultiplyVolume
            if hasattr(audio_clip, 'with_effects'):
                return audio_clip.with_effects([MultiplyVolume(volume)])
            return MultiplyVolume(volume).apply(audio_clip)
        except (ImportError, AttributeError):
            # MoviePy v1.x: use volumex method
            if hasattr(audio_clip, 'volumex'):
                return audio_clip.volumex(volume)
            # Last resort fallback
            logger.warning("Could not apply volume adjustment, returning original clip")
            return audio_clip

    def _set_audio(self, video_clip: Any, audio_clip: Any) -> Any:
        """Set audio on a video clip.

        Supports both MoviePy v1.x (set_audio) and v2.x (with_audio).

        Args:
            video_clip: Video clip to set audio on.
            audio_clip: Audio clip to attach.

        Returns:
            Video clip with audio attached.
        """
        # MoviePy v2.x uses with_audio, v1.x uses set_audio
        if hasattr(video_clip, 'with_audio'):
            return video_clip.with_audio(audio_clip)
        return video_clip.set_audio(audio_clip)

    def _audio_fadein(self, audio_clip: Any, duration: float) -> Any:
        """Apply fade-in effect to audio clip.

        Args:
            audio_clip: Audio clip to fade in.
            duration: Fade duration in seconds.

        Returns:
            Audio clip with fade-in applied.
        """
        try:
            from moviepy.audio.fx import AudioFadeIn
            if hasattr(audio_clip, 'with_effects'):
                return audio_clip.with_effects([AudioFadeIn(duration=duration)])
            return AudioFadeIn(duration=duration).apply(audio_clip)
        except (ImportError, AttributeError):
            try:
                from moviepy.audio.fx import audio_fadein
                return audio_fadein(audio_clip, duration)
            except (ImportError, AttributeError):
                if hasattr(audio_clip, 'audio_fadein'):
                    return audio_clip.audio_fadein(duration)
                # Manual fade-in using volume envelope
                return audio_clip.volumex(
                    lambda t: min(1.0, t / duration) if t < duration else 1.0
                )

    def _audio_fadeout(self, audio_clip: Any, duration: float) -> Any:
        """Apply fade-out effect to audio clip.

        Args:
            audio_clip: Audio clip to fade out.
            duration: Fade duration in seconds.

        Returns:
            Audio clip with fade-out applied.
        """
        try:
            from moviepy.audio.fx import AudioFadeOut
            if hasattr(audio_clip, 'with_effects'):
                return audio_clip.with_effects([AudioFadeOut(duration=duration)])
            return AudioFadeOut(duration=duration).apply(audio_clip)
        except (ImportError, AttributeError):
            try:
                from moviepy.audio.fx import audio_fadeout
                return audio_fadeout(audio_clip, duration)
            except (ImportError, AttributeError):
                if hasattr(audio_clip, 'audio_fadeout'):
                    return audio_clip.audio_fadeout(duration)
                # Manual fade-out using volume envelope
                clip_duration = audio_clip.duration
                return audio_clip.volumex(
                    lambda t: max(
                        0.0,
                        1.0 - (t - (clip_duration - duration)) / duration
                    ) if t > clip_duration - duration else 1.0
                )

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
        if use_frame_interpolation is None: # AB -check
            use_frame_interpolation = self._settings.video_use_frame_interpolation
        if not video_clips:
            raise CompositionError("No video clips provided", stage="input")

        logger.info("Composing video from %d clips", len(video_clips))

        try:
            from moviepy import (
                AudioFileClip,
                CompositeAudioClip,
                VideoFileClip,
                concatenate_audioclips,
                concatenate_videoclips,
            )
            # Import fade effects - handle different MoviePy versions
            try:
                # Try MoviePy 2.x style: effects as classes
                from moviepy.video.fx import FadeIn, FadeOut
                # Convert classes to functions for compatibility
                # Note: with_effects() expects a list of effects
                def fadein(clip, duration):
                    if hasattr(clip, 'with_effects'):
                        return clip.with_effects([FadeIn(duration=duration)])
                    else:
                        # Fallback: try direct instantiation and application
                        return FadeIn(duration=duration).apply(clip)
                def fadeout(clip, duration):
                    if hasattr(clip, 'with_effects'):
                        return clip.with_effects([FadeOut(duration=duration)])
                    else:
                        # Fallback: try direct instantiation and application
                        return FadeOut(duration=duration).apply(clip)
            except (ImportError, AttributeError):
                # Fallback to MoviePy 1.x style: effects as functions
                try:
                    from moviepy.video.fx import fadein, fadeout
                except ImportError:
                    # Last resort: access via module attributes
                    import moviepy.video.fx as vfx_module
                    fadein = getattr(vfx_module, 'fadein', None)
                    fadeout = getattr(vfx_module, 'fadeout', None)
                    if fadein is None or fadeout is None:
                        raise ImportError("Could not import fadein/fadeout from moviepy.video.fx")
            from moviepy.audio.fx import AudioLoop

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
                        video = self._set_audio(video, audio)

                clips.append(video)
                total_duration += video.duration

            if not clips:
                raise CompositionError("No valid video clips loaded", stage="loading")

            # Apply transitions if enabled
            if transitions and len(clips) > 1:                
                try:                
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
                                clip = fadein(clip, fade_duration)
                            if i < len(clips) - 1:
                                clip = fadeout(clip, fade_duration)
                            processed_clips.append(clip)
                        clips = processed_clips
                except Exception as e:
                    logger.warning("Failed to apply transitions: %s", e)
                    raise CompositionError(f"Transition application failed: {e}", stage="transitions") from e
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
                    # Apply AudioLoop effect - handle different MoviePy versions
                    try:
                        # MoviePy v2.x: use with_effects with list
                        if hasattr(music, 'with_effects'):
                            music = music.with_effects([AudioLoop(duration=final_video.duration)])
                        else:
                            # MoviePy v1.x: use fx method
                            music = music.fx(AudioLoop, duration=final_video.duration)
                    except (AttributeError, TypeError):
                        # Fallback: manually loop by concatenating
                        from moviepy import concatenate_audioclips
                        loops_needed = int(final_video.duration / music.duration) + 1
                        music = concatenate_audioclips([music] * loops_needed).subclip(0, final_video.duration)
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
                music_audio = self._apply_volume(music_audio, self._settings.music_volume)

                # Mix with existing audio
                if final_video.audio:
                    mixed_audio = CompositeAudioClip([final_video.audio, music_audio])
                    final_video = self._set_audio(final_video, mixed_audio)
                else:
                    final_video = self._set_audio(final_video, music_audio)

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

    def _load_and_loop_audio(
        self, audio_path: Path, target_duration: float
    ) -> Any | None:
        """Load audio file and loop or trim to target duration.

        Args:
            audio_path: Path to audio file.
            target_duration: Target duration in seconds.

        Returns:
            Audio clip (looped or trimmed) or None on failure.
        """
        from moviepy import AudioFileClip, concatenate_audioclips
        from moviepy.audio.fx import AudioLoop

        try:
            audio = AudioFileClip(str(audio_path))
            if audio.duration < target_duration:
                try:
                    if hasattr(audio, 'with_effects'):
                        audio = audio.with_effects([AudioLoop(duration=target_duration)])
                    else:
                        audio = audio.fx(AudioLoop, duration=target_duration)
                except (AttributeError, TypeError):
                    loops_needed = int(target_duration / audio.duration) + 1
                    audio = concatenate_audioclips([audio] * loops_needed).subclip(
                        0, target_duration
                    )
            else:
                audio = audio.subclip(0, target_duration)
            return audio
        except (IOError, AttributeError, TypeError) as e:
            logger.warning("Failed to load/loop audio %s: %s", audio_path, e)
            return None

    def _apply_audio_crossfades(
        self, audio_clip: Any, fade_in: bool, fade_out: bool, duration: float
    ) -> Any:
        """Apply fade in/out effects to audio clip.

        Args:
            audio_clip: Audio clip to modify.
            fade_in: Whether to apply fade-in.
            fade_out: Whether to apply fade-out.
            duration: Fade duration in seconds.

        Returns:
            Audio clip with fades applied.
        """
        if fade_in:
            audio_clip = self._audio_fadein(audio_clip, duration)
        if fade_out:
            audio_clip = self._audio_fadeout(audio_clip, duration)
        return audio_clip

    def _mix_scene_music(
        self,
        scene_tracks: list[SceneMusicTrack],
        total_duration: float,
        crossfade_duration: float = 1.0,
    ) -> Any | None:
        """Mix multiple scene music tracks with crossfades.

        Args:
            scene_tracks: List of scene music tracks with timing info.
            total_duration: Total video duration.
            crossfade_duration: Duration of crossfade between tracks.

        Returns:
            Combined audio clip with crossfaded music.
        """
        from moviepy import CompositeAudioClip

        if not scene_tracks:
            return None

        scene_tracks = sorted(scene_tracks, key=lambda x: x.start_time)
        audio_clips = []

        for i, track in enumerate(scene_tracks):
            if not track.path.exists():
                logger.warning("Scene music not found: %s", track.path)
                continue

            scene_duration = track.end_time - track.start_time
            audio = self._load_and_loop_audio(track.path, scene_duration)
            if audio is None:
                continue

            try:
                fade_in = i > 0
                fade_out = i < len(scene_tracks) - 1
                audio = self._apply_audio_crossfades(
                    audio, fade_in, fade_out, crossfade_duration
                )
                # MoviePy 2.x uses with_start(), 1.x uses set_start()
                if hasattr(audio, 'with_start'):
                    audio = audio.with_start(track.start_time)
                else:
                    audio = audio.set_start(track.start_time)
                audio_clips.append(audio)
            except Exception as e:
                logger.warning("Failed to apply audio crossfades: %s", e)
                continue

        if not audio_clips:
            return None

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
        from moviepy import ImageSequenceClip, concatenate_videoclips
        # Import fade effects - handle different MoviePy versions
        try:
            # Try MoviePy 2.x style: effects as classes
            from moviepy.video.fx import FadeIn, FadeOut
            # Convert classes to functions for compatibility
            # Note: with_effects() expects a list of effects
            def fadein(clip, duration):
                if hasattr(clip, 'with_effects'):
                    return clip.with_effects([FadeIn(duration=duration)])
                else:
                    # Fallback: try direct instantiation and application
                    return FadeIn(duration=duration).apply(clip)
            def fadeout(clip, duration):
                if hasattr(clip, 'with_effects'):
                    return clip.with_effects([FadeOut(duration=duration)])
                else:
                    # Fallback: try direct instantiation and application
                    return FadeOut(duration=duration).apply(clip)
        except (ImportError, AttributeError):
            # Fallback to MoviePy 1.x style: effects as functions
            try:
                from moviepy.video.fx import fadein, fadeout
            except ImportError:
                # Last resort: access via module attributes
                import moviepy.video.fx as vfx_module
                fadein = getattr(vfx_module, 'fadein', None)
                fadeout = getattr(vfx_module, 'fadeout', None)
                if fadein is None or fadeout is None:
                    raise ImportError("Could not import fadein/fadeout from moviepy.video.fx")

        effects = self._get_video_effects()
        num_interp_frames = self._settings.video_interpolation_frames
        
        result_clips = []
        
        for i, clip in enumerate(clips):
            # Add fade transitions as well
            if i > 0:
                clip = fadein(clip, transition_duration / 2)
            if i < len(clips) - 1:
                clip = fadeout(clip, transition_duration / 2)
            
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
        from moviepy import CompositeVideoClip, TextClip

        subtitle_clips = []

        for start_time, end_time, text in subtitles:
            try:
                txt_clip = TextClip(
                    text,
                    fontsize=self._settings.video_subtitle_fontsize,
                    color="white",
                    stroke_color="black",
                    stroke_width=2,
                    method="caption",
                    size=(video.w - 100, None),
                )
                # MoviePy 2.x uses with_* methods, 1.x uses set_* methods
                if hasattr(txt_clip, 'with_position'):
                    txt_clip = txt_clip.with_position(("center", "bottom"))
                    txt_clip = txt_clip.with_start(start_time)
                    txt_clip = txt_clip.with_duration(end_time - start_time)
                else:
                    txt_clip = txt_clip.set_position(("center", "bottom"))
                    txt_clip = txt_clip.set_start(start_time)
                    txt_clip = txt_clip.set_duration(end_time - start_time)
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
            from moviepy import VideoFileClip
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
            from moviepy import VideoFileClip

            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            clip.close()
            return duration

        except Exception as e:
            logger.warning("Failed to get video duration: %s", e)
            return 0.0
