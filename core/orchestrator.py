"""Pipeline orchestrator for video generation."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from core.models import JobStatus, ProcessedStory, SceneList, StoryInput, VideoJob

if TYPE_CHECKING:
    from config.settings import Settings, ExecutionProfile
    from core.agents.audio_tts import TTSAgent
    from core.agents.json_repair import JSONRepairAgent
    from core.agents.music_generator import MusicAgent
    from core.agents.scene_splitter import SceneSplitterAgent
    from core.agents.shot_planner import ShotPlannerAgent
    from core.agents.video_compositor import VideoCompositorAgent
    from core.agents.video_generator import VideoGenerationAgent
    from core.services.model_router import ModelRouter
    from core.services.nlp import StoryPreprocessor
    from core.services.style_context import StyleContextManager

logger = logging.getLogger(__name__)


class VideoGenerationPipeline:
    """Orchestrates the video generation pipeline.

    Coordinates all agent stages:
    0. Story preprocessing (NLP)
    1. Scene breakdown (LLM)
    2. Shot planning (LLM, optionally parallel)
    3. Video generation
    4. Audio generation (TTS + Music)
    5. Video composition

    Manages job state and provides progress tracking.
    """

    def __init__(
        self,
        *,
        settings: "Settings",
        scene_splitter: "SceneSplitterAgent",
        shot_planner: "ShotPlannerAgent",
        video_agent: "VideoGenerationAgent",
        tts_agent: "TTSAgent",
        music_agent: "MusicAgent",
        compositor: "VideoCompositorAgent",
        nlp_processor: "StoryPreprocessor | None" = None,
        style_context_manager: "StyleContextManager | None" = None,
        json_repair_agent: "JSONRepairAgent | None" = None,
        model_router: "ModelRouter | None" = None,
    ):
        """Initialize the pipeline with all required agents.

        Args:
            settings: Application settings.
            scene_splitter: Agent for splitting stories into scenes.
            shot_planner: Agent for planning shots per scene.
            video_agent: Agent for generating video clips.
            tts_agent: Agent for text-to-speech generation.
            music_agent: Agent for background music generation.
            compositor: Agent for composing final video.
            nlp_processor: Optional NLP processor for story preprocessing.
            style_context_manager: Optional style context for visual consistency.
            json_repair_agent: Optional JSON repair agent for fixing malformed responses.
            model_router: Optional model router for cost tracking and safety checks.
        """
        self._settings = settings
        self._scene_splitter = scene_splitter
        self._shot_planner = shot_planner
        self._video_agent = video_agent
        self._tts_agent = tts_agent
        self._music_agent = music_agent
        self._compositor = compositor
        self._nlp_processor = nlp_processor
        self._style_ctx = style_context_manager
        self._json_repair = json_repair_agent
        self._model_router = model_router
        self._jobs: dict[str, VideoJob] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def create_job(self, story: StoryInput) -> VideoJob:
        """Create a new video generation job.

        Args:
            story: The story input to process.

        Returns:
            A new VideoJob with pending status.
        """
        job_id = str(uuid.uuid4())
        job = VideoJob(job_id=job_id, story_input=story)
        self._jobs[job_id] = job
        self._logger.info("Created job %s for story: %s", job_id, story.title or "(untitled)")
        return job

    def get_job(self, job_id: str) -> VideoJob | None:
        """Get a job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            The VideoJob if found, None otherwise.
        """
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[VideoJob]:
        """List all jobs.

        Returns:
            List of all video jobs.
        """
        return list(self._jobs.values())

    async def execute(self, job_id: str) -> VideoJob:
        """Execute the video generation pipeline for a job.

        Args:
            job_id: The job ID to execute.ü
        Returns:
            The completed VideoJob.

        Raises:
            KeyError: If job_id is not found.
        """
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        job.status = JobStatus.PROCESSING
        self._logger.info("Starting pipeline execution for job %s", job_id)

        try:
            # Stage 0: Story Preprocessing (NLP)
            await self._stage_preprocessing(job)

            # Stage 1: Scene Breakdown
            await self._stage_scene_breakdown(job)

            # Stage 2: Shot Planning
            await self._stage_shot_planning(job)


            #if self._settings.execution_profile == "debug_cpu":
            job.scenes.scenes = job.scenes.scenes[:1]
            job.scenes.scenes[0].shots = job.scenes.scenes[0].shots[:2]

            # Stage 3: Video Generation
            await self._stage_video_generation(job)

            # Stage 4: Audio Generation
            await self._stage_audio_generation(job)

            # Stage 5: Composition
            await self._stage_composition(job)

            job.status = JobStatus.COMPLETED
            job.progress = "Complete"
            self._logger.info("Pipeline completed successfully for job %s", job_id)

            # Log cost summary if model router is available
            if self._model_router:
                cost_summary = self._model_router.get_cost_summary()
                self._logger.info(
                    "Cost summary for job %s: Total=$%.4f, By stage: %s",
                    job_id,
                    cost_summary["total_cost"],
                    cost_summary["by_stage"],
                )

        except Exception as e:
            self._logger.exception("Pipeline failed for job %s: %s", job_id, e)
            job.status = JobStatus.FAILED
            job.error_message = str(e)

        return job

    async def _stage_preprocessing(self, job: VideoJob) -> None:
        """Execute story preprocessing stage (NLP).

        Args:
            job: The job being processed.
        """
        job.current_stage = "preprocessing"
        job.progress = "Preprocessing story..."
        self._logger.info("Stage: Story preprocessing")

        if self._nlp_processor:
            processed = await self._nlp_processor.preprocess(
                job.story_input.text,
                max_tokens=self._settings.llm_max_tokens,
                skip_summarization_threshold=self._settings.llm_skip_summarization_threshold,
            )
            job.processed_story = processed
            self._logger.info(
                "Preprocessing complete: %d tokens, %d chunks, %d characters found",
                processed.token_count,
                len(processed.chunks),
                len(processed.entities.characters),
            )

            # Initialize style context with extracted entities
            if self._style_ctx:
                from core.services.style_context import StyleContextManager

                # Re-initialize with entities if we have a style context manager
                self._style_ctx = StyleContextManager(entities=processed.entities)
                
                # Update agents with the style context
                if hasattr(self._shot_planner, "set_style_context_manager"):
                    self._shot_planner.set_style_context_manager(self._style_ctx)
                if hasattr(self._video_agent, "set_style_context_manager"):
                    self._video_agent.set_style_context_manager(self._style_ctx)
                    
                self._logger.info("Style context initialized with %d characters", 
                                  len(processed.entities.characters))
        else:
            # Create basic processed story without NLP
            job.processed_story = ProcessedStory(
                original_text=job.story_input.text,
                chunks=[job.story_input.text],
                token_count=len(job.story_input.text.split()) * 2,
            )
            self._logger.info("No NLP processor, using basic preprocessing")

    async def _stage_scene_breakdown(self, job: VideoJob) -> None:
        """Execute scene breakdown stage.

        Args:
            job: The job being processed.
        """
        job.current_stage = "scene_breakdown"
        job.progress = "Splitting story into scenes..."
        self._logger.info("Stage: Scene breakdown")

        # Pass processed story to scene splitter if available
        scene_list = await self._scene_splitter.run(
            job.story_input,
            processed_story=job.processed_story,
        )
        job.scenes = scene_list

        self._logger.info("Scene breakdown complete: %d scenes", len(scene_list.scenes))

    async def _stage_shot_planning(self, job: VideoJob) -> None:
        """Execute shot planning stage.

        Args:
            job: The job being processed.
        """
        if not job.scenes:
            raise ValueError("No scenes available for shot planning")

        job.current_stage = "shot_planning"
        self._logger.info("Stage: Shot planning")

        if self._settings.llm_parallel: # AB - Check this block
            # Parallel shot planning for all scenes
            self._logger.info("Running parallel shot planning for %d scenes", len(job.scenes.scenes))
            job.progress = "Planning shots (parallel)..."
            
            tasks = [self._shot_planner.run(scene) for scene in job.scenes.scenes]
            shot_results = await asyncio.gather(*tasks)
        else:
            # Sequential shot planning
            shot_results = []
            for scene in job.scenes.scenes:
                job.progress = f"Planning shots for scene {scene.id}/{len(job.scenes.scenes)}..."
                self._logger.info("Planning shots for scene %d", scene.id)
                shots = await self._shot_planner.run(scene)
                shot_results.append(shots)

        # Assign shots to scenes
        for scene, shots in zip(job.scenes.scenes, shot_results, strict=True):
            scene.shots = shots

        total_shots = sum(len(s.shots) for s in job.scenes.scenes)
        self._logger.info("Shot planning complete: %d total shots", total_shots)

    async def _stage_video_generation(self, job: VideoJob) -> None:
        """Execute video generation stage.

        Args:
            job: The job being processed.
        """
        if not job.scenes:
            raise ValueError("No scenes available for video generation")

        job.current_stage = "video_generation"
        self._logger.info("Stage: Video generation")

        total_shots = sum(len(s.shots) for s in job.scenes.scenes)
        current_shot = 0

        for scene in job.scenes.scenes:
            # Set current scene context for style consistency
            if hasattr(self._video_agent, "set_current_scene"):
                self._video_agent.set_current_scene(scene, shot_index=0)

            for idx, shot in enumerate(scene.shots):
                current_shot += 1
                job.progress = f"Generating video: shot {current_shot}/{total_shots}..."
                self._logger.info(
                    "Generating video for scene %d, shot %d",
                    scene.id,
                    shot.id,
                )

                # Safety check before generating video # AB - Commented out for now
                # if self._model_router:
                #     is_safe = await self._model_router.check_safety(shot.description)
                #     if not is_safe:
                #         error_msg = f"Unsafe content detected in shot {shot.id}: {shot.description[:100]}"
                #         self._logger.error(error_msg)
                #         raise ValueError(error_msg)

                # Update shot index for style context
                if hasattr(self._video_agent, "set_current_scene"):
                    self._video_agent.set_current_scene(scene, shot_index=idx)

                video_path = await self._video_agent.run(shot)
                shot.video_file_path = str(video_path)

        self._logger.info("Video generation complete: %d clips", total_shots)

    async def _stage_audio_generation(self, job: VideoJob) -> None:
        """Execute audio generation stage (TTS + Music).

        Args:
            job: The job being processed.
        """
        if not job.scenes:
            raise ValueError("No scenes available for audio generation")

        job.current_stage = "audio_generation"
        self._logger.info("Stage: Audio generation")

        for scene in job.scenes.scenes:
            # Generate TTS for shots with dialogue
            for shot in scene.shots:
                if shot.dialogue or shot.subtitle_text:
                    job.progress = f"Generating TTS: scene {scene.id}, shot {shot.id}..."
                    self._logger.info(
                        "Generating TTS for scene %d, shot %d",
                        scene.id,
                        shot.id,
                    )
                    audio_path = await self._tts_agent.run(shot)
                    shot.audio_file_path = str(audio_path)

            # Generate background music for scene
            job.progress = f"Generating music for scene {scene.id}..."
            self._logger.info("Generating music for scene %d", scene.id)
            music_path = await self._music_agent.run(scene)
            scene.music_file_path = str(music_path)

        self._logger.info("Audio generation complete")

    async def _stage_composition(self, job: VideoJob) -> None:
        """Execute video composition stage.

        Args:
            job: The job being processed.
        """
        if not job.scenes:
            raise ValueError("No scenes available for composition")

        job.current_stage = "composition"
        job.progress = "Composing final video..."
        self._logger.info("Stage: Video composition")

        final_path = await self._compositor.run(job.scenes)
        job.final_video_path = str(final_path)

        self._logger.info("Composition complete: %s", final_path)
