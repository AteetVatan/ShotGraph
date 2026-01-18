"""Background task definitions for video generation.

This module provides task wrappers for running the video generation pipeline
in background threads or with task queue systems like Celery.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.orchestrator import VideoGenerationPipeline

logger = logging.getLogger(__name__)

# Thread pool for running async tasks in background
_executor = ThreadPoolExecutor(max_workers=2)


def run_pipeline_sync(pipeline: "VideoGenerationPipeline", job_id: str) -> None:
    """Run the pipeline synchronously in a background thread.

    Args:
        pipeline: The configured pipeline instance.
        job_id: The job ID to execute.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(pipeline.execute(job_id))
        finally:
            loop.close()
    except Exception as e:
        logger.exception(f"Pipeline execution failed for job {job_id}: {e}")
        # Update job status to failed
        job = pipeline.get_job(job_id)
        if job:
            from core.models import JobStatus

            job.status = JobStatus.FAILED
            job.error_message = str(e)


def submit_pipeline_task(pipeline: "VideoGenerationPipeline", job_id: str) -> None:
    """Submit pipeline execution to the thread pool.

    Args:
        pipeline: The configured pipeline instance.
        job_id: The job ID to execute.
    """
    _executor.submit(run_pipeline_sync, pipeline, job_id)
