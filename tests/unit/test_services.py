"""Unit tests for service implementations."""

import pytest

from core.constants import FieldNames
from pathlib import Path
from unittest.mock import AsyncMock, patch

from config.settings import ExecutionProfile, Settings
from core.models import Language


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Provide test settings with temp paths."""
    return Settings(
        execution_profile=ExecutionProfile.DEBUG_CPU,
        gpu_enabled=False,
        llm_api_key="test_key",
        storage_path=str(tmp_path / "output"),
        assets_path=str(tmp_path / "assets"),
        max_retries=1,
    )


class TestMockVideoGenerator:
    """Tests for MockVideoGenerator."""

    def test_generates_video_file(self, test_settings: Settings) -> None:
        """Test that mock generator creates a video file."""
        from core.services.vision import MockVideoGenerator

        generator = MockVideoGenerator(test_settings)
        result = generator.generate(
            prompt="A wide shot of a village",
            duration_seconds=2.0,
        )

        assert result.exists()
        assert result.suffix == ".mp4"

    def test_caches_repeated_requests(self, test_settings: Settings) -> None:
        """Test that identical requests return cached results."""
        from core.services.vision import MockVideoGenerator

        generator = MockVideoGenerator(test_settings)
        
        result1 = generator.generate(prompt="Same prompt", duration_seconds=2.0)
        result2 = generator.generate(prompt="Same prompt", duration_seconds=2.0)

        assert result1 == result2

    def test_different_prompts_create_different_files(self, test_settings: Settings) -> None:
        """Test that different prompts create different files."""
        from core.services.vision import MockVideoGenerator

        generator = MockVideoGenerator(test_settings)

        result1 = generator.generate(prompt="First prompt", duration_seconds=2.0)
        result2 = generator.generate(prompt="Second prompt", duration_seconds=2.0)

        assert result1 != result2


class TestMockTTSGenerator:
    """Tests for MockTTSGenerator."""

    def test_generates_audio_file(self, test_settings: Settings) -> None:
        """Test that mock TTS creates an audio file."""
        from core.services.tts import MockTTSGenerator

        generator = MockTTSGenerator(test_settings)
        result = generator.generate(
            text="Hello world, this is a test.",
            language=Language.ENGLISH,
        )

        assert result.exists()
        assert result.suffix == ".wav"

    def test_duration_scales_with_text_length(self, test_settings: Settings) -> None:
        """Test that longer text creates longer audio."""
        from core.services.tts import MockTTSGenerator
        import wave

        generator = MockTTSGenerator(test_settings)

        # Short text
        short_result = generator.generate(text="Short text.", language=Language.ENGLISH)
        with wave.open(str(short_result), "rb") as wf:
            short_duration = wf.getnframes() / wf.getframerate()

        # Long text
        long_text = " ".join(["word"] * 100)
        long_result = generator.generate(text=long_text, language=Language.ENGLISH)
        with wave.open(str(long_result), "rb") as wf:
            long_duration = wf.getnframes() / wf.getframerate()

        assert long_duration > short_duration


class TestMockMusicGenerator:
    """Tests for MockMusicGenerator."""

    def test_generates_music_file(self, test_settings: Settings) -> None:
        """Test that mock music generator creates an audio file."""
        from core.services.music import MockMusicGenerator

        generator = MockMusicGenerator(test_settings)
        result = generator.generate(
            prompt="Dramatic orchestral music",
            duration_seconds=5.0,
        )

        assert result.exists()
        assert result.suffix == ".wav"


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    @pytest.mark.asyncio
    async def test_returns_scene_response_for_scene_prompt(self) -> None:
        """Test mock returns appropriate scene breakdown."""
        from core.services.llm_client import MockLLMClient
        import json

        client = MockLLMClient()
        response = await client.complete(
            system_prompt="Split the story into scenes",
            user_prompt="Once upon a time...",
        )

        data = json.loads(response)
        assert FieldNames.SCENES in data
        assert len(data[FieldNames.SCENES]) > 0
        assert FieldNames.ID in data[FieldNames.SCENES][0]
        assert FieldNames.SUMMARY in data[FieldNames.SCENES][0]

    @pytest.mark.asyncio
    async def test_returns_shot_response_for_director_prompt(self) -> None:
        """Test mock returns appropriate shot planning."""
        from core.services.llm_client import MockLLMClient
        import json

        client = MockLLMClient()
        response = await client.complete(
            system_prompt="You are a film director",
            user_prompt="Create shots for this scene",
        )

        data = json.loads(response)
        assert FieldNames.SHOTS in data
        assert len(data[FieldNames.SHOTS]) > 0
        assert FieldNames.ID in data[FieldNames.SHOTS][0]
        assert FieldNames.DESCRIPTION in data[FieldNames.SHOTS][0]


class TestTogetherLLMClient:
    """Tests for TogetherLLMClient."""

    @pytest.mark.asyncio
    async def test_raises_without_api_key(self, test_settings: Settings) -> None:
        """Test that missing API key raises error."""
        from core.services.llm_client import TogetherLLMClient, LLMClientError

        test_settings.llm_api_key = ""
        client = TogetherLLMClient(test_settings)

        with pytest.raises(LLMClientError, match="API key not configured"):
            await client.complete(
                system_prompt="Test",
                user_prompt="Test",
            )

    @pytest.mark.asyncio
    async def test_makes_correct_api_call(self, test_settings: Settings) -> None:
        """Test that API call is structured correctly."""
        from core.services.llm_client import TogetherLLMClient
        import httpx

        client = TogetherLLMClient(test_settings)

        mock_response = {
            "choices": [{"message": {"content": "Test response"}}]
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                status_code=200,
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )

            result = await client.complete(
                system_prompt="System",
                user_prompt="User",
                temperature=0.5,
            )

            assert result == "Test response"
            mock_post.assert_called_once()
