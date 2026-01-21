"""Pydantic Settings configuration for ShotGraph."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionProfile(str, Enum):
    """Execution profile for the pipeline."""

    DEBUG_CPU = "debug_cpu"
    PROD_GPU = "prod_gpu"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Execution
    execution_profile: ExecutionProfile = Field(
        default=ExecutionProfile.DEBUG_CPU,
        description="Execution profile (debug_cpu or prod_gpu)",
    )
    gpu_enabled: bool = Field(default=False, description="Whether GPU is enabled")

    # LLM Configuration
    llm_provider: str = Field(
        default="together",
        description="LLM provider (together, local)",
    )
    llm_api_key: str = Field(default="", description="LLM API key")
    llm_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        description="LLM model identifier",
    )
    llm_parallel: bool = Field(
        default=False,
        description="Enable parallel LLM calls for shot planning",
    )
    llm_together_base_url: str = Field(
        default="https://api.together.xyz/v1",
        description="Together.ai API base URL",
    )
    llm_together_timeout: float = Field(
        default=120.0,
        ge=1.0,
        le=600.0,
        description="Together.ai API request timeout in seconds",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="Maximum tokens for LLM responses",
    )

    # Per-Stage Model Configuration (Cost-Optimized Routing)
    # Step A - Story compression/beat extraction (cheapest)
    llm_model_story_compress: str = Field(
        default="google/gemma-3n-E4B-it",
        description="Model for story compression (Step A) - $0.02/$0.04 per 1M tokens",
    )
    llm_model_story_compress_fallback: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        description="Fallback for Step A - $0.06/$0.06 per 1M tokens",
    )

    # Step B - Scene breakdown draft (output-heavy, keep output cheap)
    llm_model_scene_draft: str = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        description="Model for scene breakdown (Step B) - $0.18/$0.18 per 1M tokens",
    )
    llm_model_scene_draft_large: str = Field(
        default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        description="Large context model for Step B (when needed) - $0.18/$0.59 per 1M tokens",
    )

    # Step C - Shot plan finalization (input can get big)
    llm_model_shot_final: str = Field(
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        description="Model for shot planning (Step C) - $0.27/$0.85 per 1M tokens",
    )
    llm_model_shot_final_fallback: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        description="Fallback for Step C (hard cases) - $0.88/$0.88 per 1M tokens",
    )

    # Step D - JSON validation/repair (small + structured outputs)
    llm_model_json_repair: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        description="Model for JSON validation/repair (Step D) - $0.06/$0.06 per 1M tokens",
    )

    # Safety/moderation
    llm_safety_model: str = Field(
        default="meta-llama/Llama-Guard-4-12B",
        description="Safety/moderation model",
    )

    # Cost control thresholds
    llm_skip_summarization_threshold: int = Field(
        default=2000,
        ge=0,
        description="Skip NLP summarization if story < N tokens (saves LLM cost)",
    )
    llm_use_large_context_threshold: int = Field(
        default=8000,
        ge=0,
        description="Use large context model if input > N tokens",
    )

    # Video Generation
    video_model: str = Field(
        default="stable-video-diffusion",
        description="Video generation model",
    )
    video_model_path: str = Field(default="", description="Path to video model weights")
    video_resolution: str = Field(default="1024x576", description="Output video resolution")
    video_fps: int = Field(default=24, ge=1, le=60, description="Output video FPS")
    default_shot_duration: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Default shot duration in seconds",
    )
    video_codec: str = Field(default="libx264", description="Video codec for encoding")
    video_audio_codec: str = Field(default="aac", description="Audio codec for encoding")
    video_transition_duration: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Fade transition duration in seconds",
    )
    video_subtitle_fontsize: int = Field(
        default=36,
        ge=12,
        le=72,
        description="Subtitle font size in pixels",
    )
    video_use_frame_interpolation: bool = Field(
        default=False,
        description="Use frame interpolation for smoother transitions",
    )
    video_interpolation_frames: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of frames to interpolate between shots",
    )
    video_ken_burns_enabled: bool = Field(
        default=True,
        description="Enable Ken Burns (pan/zoom) effect for still images",
    )
    video_ken_burns_zoom_range: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Maximum zoom amount for Ken Burns effect (0.2 = 20%)",
    )

    # TTS Configuration
    tts_voice_en: str = Field(
        default="en-US-AriaNeural",
        description="English voice ID for Edge TTS (e.g., en-US-AriaNeural, en-US-JennyNeural)",
    )
    tts_voice_hi: str = Field(
        default="hi-IN-SwaraNeural",
        description="Hindi voice ID for Edge TTS (e.g., hi-IN-SwaraNeural, hi-IN-MadhurNeural)",
    )

    # Music Generation
    music_model: str = Field(default="musicgen-medium", description="Music generation model")
    music_model_org: str = Field(
        default="facebook",
        description="HuggingFace organization/prefix for music model (e.g., 'facebook' for facebook/musicgen-medium)",
    )
    music_max_segment_duration: float = Field(
        default=12.0,
        ge=1.0,
        le=30.0,
        description="Maximum duration per music generation segment in seconds",
    )
    music_volume: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Background music volume (0.0-1.0)",
    )
    music_duck_level: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Music volume during narration (ducking level, 0.0-1.0)",
    )
    music_crossfade_duration: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Duration of crossfade between scene music tracks in seconds",
    )

    # Storage
    storage_path: str = Field(default="./output", description="Output storage path")
    assets_path: str = Field(default="./assets", description="Assets directory path")

    # Retry Configuration
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts per agent",
    )

    # Experimental Features
    use_toon_format: bool = Field(
        default=True,
        description="Use TOON format for LLM communication (saves ~40% tokens)",
    )

    # Security
    api_key_enabled: bool = Field(
        default=False,
        description="Enable API key authentication",
    )
    api_key: str = Field(
        default="",
        description="API key for authentication (required if api_key_enabled=True)",
    )
    rate_limit_per_minute: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum requests per minute per client",
    )
    max_story_length: int = Field(
        default=100_000,
        ge=100,
        le=1_000_000,
        description="Maximum story text length in characters",
    )
    max_story_size_bytes: int = Field(
        default=500_000,
        ge=1000,
        le=10_000_000,
        description="Maximum story size in bytes (UTF-8 encoded)",
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins (empty = no CORS)",
    )
