"""Debug script to start FastAPI server and test video generation endpoint.

This script:
1. Starts the FastAPI server from app.main
2. Directly calls the generate_video endpoint function with test data
"""

import asyncio
import logging
import sys
import threading
from pathlib import Path

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import uvicorn
from fastapi import BackgroundTasks

from app.dependencies import get_pipeline, get_settings
from app.main import generate_video
from app.schemas import GenerateRequest
from core.models import Language

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Sample test story (meets validation requirements: min 50 chars, min 10 words)
SAMPLE_STORY = (
    "Once upon a time in a small village nestled between rolling hills, "
    "there lived a young hero named Alex. Alex was known throughout the land "
    "for their courage and kindness. One day, a mysterious shadow appeared over "
    "the village, casting fear into the hearts of all who lived there. "
    "Determined to help, Alex embarked on an epic journey to discover the "
    "source of this darkness and restore peace to their home."
)

SAMPLE_TITLE = "A Heros's story"
SAMPLE_LANGUAGE = Language.ENGLISH


async def get_api_key_for_debug() -> str:
    """Get API key for debug mode.

    Returns the configured API key if auth is enabled,
    or "anonymous" if auth is disabled.

    Returns:
        API key string for use in direct function calls.
    """
    settings = get_settings()
    if not settings.api_key_enabled:
        return "anonymous"
    return settings.api_key or "anonymous"


async def test_generate_video() -> None:
    """Test the generate_video endpoint function directly.

    This function calls the generate_video endpoint handler directly
    without going through HTTP, simulating what would happen when
    the /generate endpoint is called.
    """
    logger.info("=" * 60)
    logger.info("Testing generate_video endpoint directly")
    logger.info("=" * 60)

    try:
        # Create test request
        request = GenerateRequest(
            story=SAMPLE_STORY,
            title=SAMPLE_TITLE,
            language=SAMPLE_LANGUAGE,
        )
        logger.info("Created GenerateRequest:")
        logger.info("  Title: %s", request.title)
        logger.info("  Language: %s", request.language.value)
        logger.info("  Story length: %d characters", len(request.story))

        # Get dependencies (same as FastAPI would inject)
        pipeline = get_pipeline()
        background_tasks = BackgroundTasks()
        api_key = await get_api_key_for_debug()

        logger.info("  API Key Auth: %s", "enabled" if get_settings().api_key_enabled else "disabled")

        # Call the endpoint function directly
        logger.info("\nCalling generate_video function...")
        response = await generate_video(
            request=request,
            background_tasks=background_tasks,
            pipeline=pipeline,
            _api_key=api_key,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Response received:")
        logger.info("  Job ID: %s", response.job_id)
        logger.info("  Status: %s", response.status.value)
        logger.info("  Message: %s", response.message)
        logger.info("=" * 60)

        # Execute background tasks (normally FastAPI does this)
        logger.info("\nExecuting background tasks...")
        await background_tasks()

        logger.info("\nTest completed successfully!")

    except Exception as e:
        logger.exception("Error during test: %s", e)
        raise


def run_server() -> None:
    """Start the FastAPI server using uvicorn.

    The server will run on http://localhost:8000 by default.
    """
    logger.info("Starting FastAPI server...")
    logger.info("Server will be available at: http://localhost:8000")
    logger.info("API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


async def main() -> None:
    """Main entry point for debug script.

    This function:
    1. Starts the FastAPI server in a background thread
    2. Waits a moment for server to initialize
    3. Calls the generate_video function directly
    """
    logger.info("ShotGraph Debug Script")
    logger.info("=" * 60)

    # Start FastAPI server in background thread
    logger.info("\n[Step 1] Starting FastAPI server in background...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("Server starting in background thread...")
    logger.info("Server will be available at: http://localhost:8000")
    logger.info("API docs will be available at: http://localhost:8000/docs")

    # Wait a moment for server to initialize
    logger.info("\nWaiting for server to initialize...")
    await asyncio.sleep(2)

    # Test the function directly
    logger.info("\n[Step 2] Testing generate_video function directly...")
    await test_generate_video()

    logger.info("\n" + "=" * 60)
    logger.info("Debug script completed!")
    logger.info("Server is still running in background.")
    logger.info("Press Ctrl+C to stop the server.")
    logger.info("=" * 60)

    # Keep script running so server stays alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
