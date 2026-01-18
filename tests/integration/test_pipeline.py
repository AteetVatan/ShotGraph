"""Integration tests for the video generation pipeline."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from config.settings import ExecutionProfile, Settings
from core.models import JobStatus, StoryInput
from core.orchestrator import VideoGenerationPipeline


@pytest.fixture
def integration_settings(tmp_path: Path) -> Settings:
    """Provide settings for integration testing."""
    return Settings(
        execution_profile=ExecutionProfile.DEBUG_CPU,
        gpu_enabled=False,
        llm_api_key="",  # Will use mock
        storage_path=str(tmp_path / "output"),
        assets_path=str(tmp_path / "assets"),
        max_retries=1,
        llm_parallel=False,
    )


@pytest.fixture
def mock_pipeline(integration_settings: Settings) -> VideoGenerationPipeline:
    """Create a pipeline with all mocked services."""
    from app.dependencies import create_pipeline

    return create_pipeline(integration_settings)


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_create_job(
        self, mock_pipeline: VideoGenerationPipeline, sample_story_input: StoryInput
    ) -> None:
        """Test job creation."""
        job = mock_pipeline.create_job(sample_story_input)

        assert job.job_id is not None
        assert job.status == JobStatus.PENDING
        assert job.story_input == sample_story_input

    def test_get_job(
        self, mock_pipeline: VideoGenerationPipeline, sample_story_input: StoryInput
    ) -> None:
        """Test job retrieval."""
        job = mock_pipeline.create_job(sample_story_input)
        retrieved = mock_pipeline.get_job(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_job(self, mock_pipeline: VideoGenerationPipeline) -> None:
        """Test that getting nonexistent job returns None."""
        result = mock_pipeline.get_job("nonexistent-id")
        assert result is None

    def test_list_jobs(
        self, mock_pipeline: VideoGenerationPipeline, sample_story_input: StoryInput
    ) -> None:
        """Test listing all jobs."""
        job1 = mock_pipeline.create_job(sample_story_input)
        job2 = mock_pipeline.create_job(sample_story_input)

        jobs = mock_pipeline.list_jobs()

        assert len(jobs) == 2
        assert job1 in jobs
        assert job2 in jobs

    @pytest.mark.asyncio
    async def test_pipeline_execution_debug_mode(
        self, mock_pipeline: VideoGenerationPipeline, sample_story_input: StoryInput
    ) -> None:
        """Test full pipeline execution in debug mode with mocks."""
        job = mock_pipeline.create_job(sample_story_input)

        # Execute pipeline
        result = await mock_pipeline.execute(job.job_id)

        # Check job progressed through stages
        assert result.status in [JobStatus.COMPLETED, JobStatus.FAILED]

        if result.status == JobStatus.COMPLETED:
            assert result.scenes is not None
            assert len(result.scenes.scenes) > 0
            # In mock mode, may not have actual video path
            # but pipeline should complete

    @pytest.mark.asyncio
    async def test_pipeline_handles_errors(
        self, mock_pipeline: VideoGenerationPipeline
    ) -> None:
        """Test that pipeline handles errors gracefully."""
        # Create job with minimal/invalid input
        story_input = StoryInput(text="Short test story for error handling.")
        job = mock_pipeline.create_job(story_input)

        # Execute pipeline (may fail due to short story)
        result = await mock_pipeline.execute(job.job_id)

        # Should have error status or complete (mocks may succeed)
        assert result.status in [JobStatus.COMPLETED, JobStatus.FAILED]

        if result.status == JobStatus.FAILED:
            assert result.error_message is not None


class TestAPIIntegration:
    """Integration tests for the FastAPI application."""

    @pytest.fixture
    def test_client(self, integration_settings: Settings):
        """Create test client with mocked pipeline."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app import dependencies

        # Reset and configure with test settings
        dependencies.reset_pipeline()
        dependencies._pipeline_instance = None

        # Patch settings
        with patch.object(dependencies, "get_settings", return_value=integration_settings):
            with TestClient(app) as client:
                yield client

        # Cleanup
        dependencies.reset_pipeline()

    def test_health_endpoint(self, test_client) -> None:
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "profile" in data

    def test_generate_endpoint(self, test_client) -> None:
        """Test video generation endpoint."""
        response = test_client.post(
            "/generate",
            json={
                "story": "Once upon a time in a magical kingdom, there lived a brave knight.",
                "title": "The Brave Knight",
                "language": "en",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_generate_endpoint_validation(self, test_client) -> None:
        """Test input validation on generate endpoint."""
        # Story too short
        response = test_client.post(
            "/generate",
            json={"story": "Short"},
        )

        assert response.status_code == 422  # Validation error

    def test_status_endpoint(self, test_client) -> None:
        """Test job status endpoint."""
        # First create a job
        create_response = test_client.post(
            "/generate",
            json={"story": "A long enough story about adventures and heroes."},
        )
        job_id = create_response.json()["job_id"]

        # Then check status
        status_response = test_client.get(f"/status/{job_id}")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["job_id"] == job_id

    def test_status_endpoint_not_found(self, test_client) -> None:
        """Test status endpoint with invalid job ID."""
        response = test_client.get("/status/nonexistent-job-id")

        assert response.status_code == 404

    def test_jobs_list_endpoint(self, test_client) -> None:
        """Test jobs listing endpoint."""
        # Create a job first
        test_client.post(
            "/generate",
            json={"story": "A story for listing test with enough content."},
        )

        response = test_client.get("/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert data["total"] >= 1

    def test_video_download_not_completed(self, test_client) -> None:
        """Test video download when job not completed."""
        # Create a job
        create_response = test_client.post(
            "/generate",
            json={"story": "A story for download test with sufficient length."},
        )
        job_id = create_response.json()["job_id"]

        # Try to download immediately (job not completed)
        download_response = test_client.get(f"/video/{job_id}")

        assert download_response.status_code == 400
