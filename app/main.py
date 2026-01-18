"""FastAPI application main module."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.dependencies import get_pipeline, get_settings, reset_pipeline
from app.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobListResponse,
    StatusResponse,
)
from app.security import setup_rate_limiter, verify_api_key
from config.settings import Settings
from core.models import JobStatus, StoryInput
from core.orchestrator import VideoGenerationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    settings = get_settings()
    logger.info("Starting ShotGraph API")
    logger.info("Execution Profile: %s", settings.execution_profile.value)
    logger.info("LLM Provider: %s", settings.llm_provider)
    logger.info("Storage Path: %s", settings.storage_path)
    logger.info("API Key Auth: %s", "enabled" if settings.api_key_enabled else "disabled")
    logger.info("Rate Limit: %d/minute", settings.rate_limit_per_minute)

    # Pre-initialize pipeline
    _ = get_pipeline()
    logger.info("Pipeline initialized")

    yield

    # Shutdown
    logger.info("Shutting down ShotGraph API")
    reset_pipeline()


app = FastAPI(
    title="ShotGraph",
    description="AI Cinematic Video Generation Pipeline - Convert stories to videos using AI",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Setup rate limiting
setup_rate_limiter(app)

# Setup CORS if configured
_settings = get_settings()
if _settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s", _settings.cors_origins)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "ShotGraph API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Check API health status.

    Returns the current service status, version, and execution profile.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        profile=settings.execution_profile.value,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Video Generation"])
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    pipeline: VideoGenerationPipeline = Depends(get_pipeline),
    _api_key: str = Depends(verify_api_key),
) -> GenerateResponse:
    """Start a new video generation job.

    Accepts a story text and queues it for video generation.
    Returns immediately with a job ID that can be used to track progress.

    Requires API key authentication if enabled in settings.

    Args:
        request: The generation request containing story and options.
        background_tasks: FastAPI background tasks handler.
        pipeline: The video generation pipeline (injected).
        _api_key: Validated API key (injected).

    Returns:
        Job ID and initial status.
    """
    logger.info("Received generation request: %s", request.title or "(untitled)")

    # Create story input
    story_input = StoryInput(
        text=request.story,
        title=request.title,
        language=request.language,
    )

    # Create job
    job = pipeline.create_job(story_input)
    logger.info("Created job: %s", job.job_id)

    # Queue pipeline execution in background
    background_tasks.add_task(_run_pipeline, pipeline, job.job_id)

    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        message="Video generation started. Use /status/{job_id} to track progress.",
    )


async def _run_pipeline(pipeline: VideoGenerationPipeline, job_id: str) -> None:
    """Run the pipeline in the background.

    Args:
        pipeline: The pipeline instance.
        job_id: The job to execute.
    """
    try:
        await pipeline.execute(job_id)
    except Exception as e:
        logger.exception("Background pipeline execution failed: %s", e)


@app.get("/status/{job_id}", response_model=StatusResponse, tags=["Video Generation"])
async def get_job_status(
    job_id: str,
    pipeline: VideoGenerationPipeline = Depends(get_pipeline),
) -> StatusResponse:
    """Get the status of a video generation job.

    Args:
        job_id: The job identifier.
        pipeline: The video generation pipeline (injected).

    Returns:
        Current job status and progress.

    Raises:
        HTTPException: If job is not found.
    """
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        final_video_path=job.final_video_path,
        error_message=job.error_message,
    )


@app.get("/jobs", response_model=JobListResponse, tags=["Video Generation"])
async def list_jobs(
    pipeline: VideoGenerationPipeline = Depends(get_pipeline),
) -> JobListResponse:
    """List all video generation jobs.

    Args:
        pipeline: The video generation pipeline (injected).

    Returns:
        List of all jobs with their status.
    """
    jobs = pipeline.list_jobs()
    return JobListResponse(
        jobs=[
            StatusResponse(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                current_stage=job.current_stage,
                final_video_path=job.final_video_path,
                error_message=job.error_message,
            )
            for job in jobs
        ],
        total=len(jobs),
    )


@app.get("/video/{job_id}", tags=["Video Generation"])
async def download_video(
    job_id: str,
    pipeline: VideoGenerationPipeline = Depends(get_pipeline),
) -> FileResponse:
    """Download the generated video for a completed job.

    Args:
        job_id: The job identifier.
        pipeline: The video generation pipeline (injected).

    Returns:
        The video file as a downloadable response.

    Raises:
        HTTPException: If job is not found, not completed, or video is missing.
    """
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status.value}",
        )

    if not job.final_video_path:
        raise HTTPException(
            status_code=500,
            detail="Video path is missing from completed job",
        )

    return FileResponse(
        path=job.final_video_path,
        media_type="video/mp4",
        filename=f"shotgraph_{job_id[:8]}.mp4",
    )


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.exception("Unhandled exception: %s", exc)
    return {"detail": "An unexpected error occurred", "error": str(exc)}, 500
