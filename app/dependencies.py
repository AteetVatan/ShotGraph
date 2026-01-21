"""Dependency injection for FastAPI application."""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from config.settings import ExecutionProfile, Settings
from core.agents.audio_tts import TTSAgent
from core.agents.json_repair import JSONRepairAgent
from core.agents.music_generator import MusicAgent
from core.agents.scene_splitter import SceneSplitterAgent
from core.agents.shot_planner import ShotPlannerAgent
from core.agents.video_compositor import VideoCompositorAgent
from core.agents.video_generator import VideoGenerationAgent
from core.orchestrator import VideoGenerationPipeline
from core.services.llm_client import MockLLMClient, TogetherLLMClient
from core.services.model_router import ModelRouter
from core.services.music import MockMusicGenerator, MusicGenGenerator
from core.services.tts import EdgeTTSGenerator, MockTTSGenerator
from core.services.nlp import MockNLPProcessor, StoryPreprocessor
from core.services.style_context import MockStyleContextManager, StyleContextManager
from core.services.vision import DiffuseVideoGenerator, MockVideoGenerator

if TYPE_CHECKING:
    from core.protocols.llm_client import ILLMClient
    from core.protocols.music_generator import IMusicGenerator
    from core.protocols.nlp_processor import INLPProcessor
    from core.protocols.tts_generator import ITTSGenerator
    from core.protocols.video_generator import IVideoGenerator

logger = logging.getLogger(__name__)

# Singleton pipeline instance
_pipeline_instance: VideoGenerationPipeline | None = None


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Application settings loaded from environment.
    """
    return Settings()


def detect_gpu() -> bool:
    """Detect if GPU (CUDA) is available.

    Returns:
        True if GPU is available, False otherwise.
    """
    try:
        import torch

        is_available = torch.cuda.is_available()
        if is_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s", device_name)
        return is_available
    except ImportError:
        logger.debug("PyTorch not installed, GPU detection skipped")
        return False
    except Exception as e:
        logger.warning("GPU detection failed: %s", e)
        return False


def _create_llm_client(settings: Settings) -> "ILLMClient":
    """Create the appropriate LLM client based on settings.

    Args:
        settings: Application settings.

    Returns:
        Configured LLM client instance.
    """
    if settings.execution_profile == ExecutionProfile.DEBUG_CPU and not settings.llm_api_key:
        logger.info("Using MockLLMClient (no API key configured)")
        return MockLLMClient(settings)

    provider = settings.llm_provider.lower()
    if provider == "together":
        logger.info("Using TogetherLLMClient")
        return TogetherLLMClient(settings)
    else:
        logger.info("Using MockLLMClient (unknown provider: %s)", provider)
        return MockLLMClient(settings)


def _create_video_generator(settings: Settings, is_debug: bool) -> "IVideoGenerator":
    """Create the appropriate video generator based on profile.

    Args:
        settings: Application settings.
        is_debug: Whether running in debug mode.

    Returns:
        Configured video generator instance.
    """
    if is_debug:
        logger.info("Using MockVideoGenerator")
        return MockVideoGenerator(settings)
    else:
        logger.info("Using DiffuseVideoGenerator")
        return DiffuseVideoGenerator(settings)


def _create_tts_generator(settings: Settings, is_debug: bool) -> "ITTSGenerator":
    """Create the appropriate TTS generator based on profile.

    Args:
        settings: Application settings.
        is_debug: Whether running in debug mode.

    Returns:
        Configured TTS generator instance.
    """
    if is_debug:
        logger.info("Using MockTTSGenerator")
        return MockTTSGenerator(settings)
    else:
        logger.info("Using EdgeTTSGenerator")
        return EdgeTTSGenerator(settings)


def _create_music_generator(settings: Settings, is_debug: bool) -> "IMusicGenerator":
    """Create the appropriate music generator based on profile.

    Args:
        settings: Application settings.
        is_debug: Whether running in debug mode.

    Returns:
        Configured music generator instance.
    """
    if is_debug:
        logger.info("Using MockMusicGenerator")
        return MockMusicGenerator(settings)
    else:
        logger.info("Using MusicGenGenerator")
        return MusicGenGenerator(settings)


def _create_model_router(
    settings: Settings,
    llm_client: "ILLMClient",
) -> ModelRouter | None:
    """Create the model router for cost-optimized routing.

    Args:
        settings: Application settings.
        llm_client: LLM client instance.

    Returns:
        ModelRouter instance, or None if in debug mode without API key.
    """
    if settings.execution_profile == ExecutionProfile.DEBUG_CPU and not settings.llm_api_key:
        logger.info("Skipping ModelRouter (debug mode, no API key)")
        return None

    logger.info("Creating ModelRouter for cost-optimized routing")
    return ModelRouter(settings=settings, llm_client=llm_client)


def _create_nlp_processor(
    settings: Settings,
    is_debug: bool,
    model_router: ModelRouter | None = None,
) -> "INLPProcessor":
    """Create the appropriate NLP processor based on profile.

    Args:
        settings: Application settings.
        is_debug: Whether running in debug mode.
        model_router: Optional model router for cost-optimized summarization.

    Returns:
        Configured NLP processor instance.
    """
    if is_debug:
        logger.info("Using MockNLPProcessor")
        return MockNLPProcessor()
    else:
        logger.info("Using StoryPreprocessor with spaCy")
        return StoryPreprocessor(
            model_router=model_router,
            use_spacy=True,
        )


def _create_style_context_manager(is_debug: bool) -> StyleContextManager | MockStyleContextManager:
    """Create the appropriate style context manager based on profile.

    Args:
        is_debug: Whether running in debug mode.

    Returns:
        Configured style context manager instance.
    """
    if is_debug:
        logger.info("Using MockStyleContextManager")
        return MockStyleContextManager()
    else:
        logger.info("Using StyleContextManager")
        return StyleContextManager()


def create_pipeline(settings: Settings | None = None) -> VideoGenerationPipeline:
    """Create a new pipeline instance with injected dependencies.

    Args:
        settings: Optional settings override.

    Returns:
        Configured VideoGenerationPipeline instance.
    """
    settings = settings or get_settings()

    # Auto-detect GPU and adjust profile if needed
    if settings.execution_profile == ExecutionProfile.DEBUG_CPU:
        if detect_gpu() and settings.gpu_enabled:
            logger.info("GPU detected, switching to PROD_GPU profile")
            settings.execution_profile = ExecutionProfile.PROD_GPU

    is_debug = settings.execution_profile == ExecutionProfile.DEBUG_CPU
    logger.info(
        "Creating pipeline with profile: %s (debug=%s)",
        settings.execution_profile.value,
        is_debug,
    )

    # Create service instances
    llm_client = _create_llm_client(settings)
    model_router = _create_model_router(settings, llm_client)
    video_gen = _create_video_generator(settings, is_debug)
    tts_gen = _create_tts_generator(settings, is_debug)
    music_gen = _create_music_generator(settings, is_debug)
    nlp_processor = _create_nlp_processor(settings, is_debug, model_router)
    style_ctx_manager = _create_style_context_manager(is_debug)

    # Create agents with injected services
    # Note: Agents will be updated in Phase 5 to use model_router instead of llm_client
    # For now, we pass both for backward compatibility during migration
    scene_splitter = SceneSplitterAgent(
        llm_client=llm_client,
        model_router=model_router,
        max_retries=settings.max_retries,
        settings=settings,
    )
    shot_planner = ShotPlannerAgent(
        llm_client=llm_client,
        model_router=model_router,
        max_retries=settings.max_retries,
        style_context_manager=style_ctx_manager,
        settings=settings,
    )
    video_agent = VideoGenerationAgent(
        video_generator=video_gen,
        max_retries=settings.max_retries,
        style_context_manager=style_ctx_manager,
    )
    tts_agent = TTSAgent(
        tts_generator=tts_gen,
        max_retries=settings.max_retries,
    )
    music_agent = MusicAgent(
        music_generator=music_gen,
        max_retries=settings.max_retries,
    )
    compositor = VideoCompositorAgent(settings=settings)

    # Create JSON repair agent if model router is available
    json_repair = None
    if model_router:
        json_repair = JSONRepairAgent(
            model_router=model_router,
            max_retries=1,
        )

    return VideoGenerationPipeline(
        settings=settings,
        scene_splitter=scene_splitter,
        shot_planner=shot_planner,
        video_agent=video_agent,
        tts_agent=tts_agent,
        music_agent=music_agent,
        compositor=compositor,
        nlp_processor=nlp_processor,
        style_context_manager=style_ctx_manager,
        json_repair_agent=json_repair,
        model_router=model_router,
    )


def get_pipeline() -> VideoGenerationPipeline:
    """Get the singleton pipeline instance.

    Creates a new instance if one doesn't exist.

    Returns:
        The pipeline instance.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = create_pipeline()
    return _pipeline_instance


def reset_pipeline() -> None:
    """Reset the singleton pipeline instance.

    Useful for testing or reconfiguration.
    """
    global _pipeline_instance
    _pipeline_instance = None
    logger.info("Pipeline instance reset")
