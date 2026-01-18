"""Unit tests for agent classes."""

import pytest
from unittest.mock import AsyncMock

from core.agents.base import BaseAgent
from core.agents.scene_splitter import SceneSplitterAgent
from core.agents.shot_planner import ShotPlannerAgent
from core.exceptions import LLMParseError, RetryExhaustedError
from core.models import Scene, SceneList, Shot, StoryInput


class ConcreteTestAgent(BaseAgent[StoryInput, str]):
    """Concrete implementation for testing BaseAgent."""

    def __init__(self, *, max_retries: int = 2, should_fail: int = 0):
        super().__init__(max_retries=max_retries)
        self.should_fail = should_fail
        self.attempt_count = 0

    async def _execute(self, input_data: StoryInput) -> str:
        self.attempt_count += 1
        if self.attempt_count <= self.should_fail:
            raise ValueError(f"Simulated failure {self.attempt_count}")
        return f"Success: {input_data.title}"


class TestBaseAgent:
    """Tests for BaseAgent class."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, sample_story_input: StoryInput) -> None:
        """Test successful execution on first attempt."""
        agent = ConcreteTestAgent(max_retries=2, should_fail=0)
        result = await agent.run(sample_story_input)
        assert result == f"Success: {sample_story_input.title}"
        assert agent.attempt_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self, sample_story_input: StoryInput) -> None:
        """Test retry leading to success."""
        agent = ConcreteTestAgent(max_retries=2, should_fail=1)
        result = await agent.run(sample_story_input)
        assert result == f"Success: {sample_story_input.title}"
        assert agent.attempt_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, sample_story_input: StoryInput) -> None:
        """Test that RetryExhaustedError is raised after all retries fail."""
        agent = ConcreteTestAgent(max_retries=2, should_fail=5)  # Always fail
        with pytest.raises(RetryExhaustedError) as exc_info:
            await agent.run(sample_story_input)
        assert agent.attempt_count == 3  # 1 initial + 2 retries
        assert "3 attempts failed" in str(exc_info.value)


class TestSceneSplitterAgent:
    """Tests for SceneSplitterAgent."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create a mock LLM client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_successful_scene_split(
        self, mock_llm: AsyncMock, sample_story_input: StoryInput
    ) -> None:
        """Test successful scene splitting."""
        mock_llm.complete.return_value = """{
            "scenes": [
                {"id": 1, "summary": "Hero in village", "text": "Once upon a time..."},
                {"id": 2, "summary": "Meeting wizard", "text": "One morning..."}
            ]
        }"""

        agent = SceneSplitterAgent(llm_client=mock_llm, max_retries=1)
        result = await agent.run(sample_story_input)

        assert isinstance(result, SceneList)
        assert len(result.scenes) == 2
        assert result.scenes[0].id == 1
        assert result.scenes[0].summary == "Hero in village"

    @pytest.mark.asyncio
    async def test_handles_markdown_code_block(
        self, mock_llm: AsyncMock, sample_story_input: StoryInput
    ) -> None:
        """Test handling of markdown code blocks in response."""
        mock_llm.complete.return_value = """Here is the breakdown:

```json
{
    "scenes": [
        {"id": 1, "summary": "Scene one", "text": "Text..."}
    ]
}
```"""

        agent = SceneSplitterAgent(llm_client=mock_llm, max_retries=1)
        result = await agent.run(sample_story_input)

        assert len(result.scenes) == 1

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(
        self, mock_llm: AsyncMock, sample_story_input: StoryInput
    ) -> None:
        """Test that invalid JSON raises LLMParseError."""
        mock_llm.complete.return_value = "This is not valid JSON"

        agent = SceneSplitterAgent(llm_client=mock_llm, max_retries=0)
        with pytest.raises(RetryExhaustedError):
            await agent.run(sample_story_input)

    @pytest.mark.asyncio
    async def test_raises_on_empty_scenes(
        self, mock_llm: AsyncMock, sample_story_input: StoryInput
    ) -> None:
        """Test that empty scenes list raises error."""
        mock_llm.complete.return_value = '{"scenes": []}'

        agent = SceneSplitterAgent(llm_client=mock_llm, max_retries=0)
        with pytest.raises(RetryExhaustedError):
            await agent.run(sample_story_input)


class TestShotPlannerAgent:
    """Tests for ShotPlannerAgent."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create a mock LLM client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_successful_shot_planning(
        self, mock_llm: AsyncMock, sample_scene: Scene
    ) -> None:
        """Test successful shot planning."""
        mock_llm.complete.return_value = """{
            "shots": [
                {
                    "id": 1,
                    "description": "Wide shot of village well",
                    "duration": 5,
                    "shot_type": "wide",
                    "dialogue": null
                },
                {
                    "id": 2,
                    "description": "Close-up of wizard's face",
                    "duration": 7,
                    "shot_type": "close_up",
                    "dialogue": "There is a dragon..."
                }
            ]
        }"""

        agent = ShotPlannerAgent(llm_client=mock_llm, max_retries=1)
        result = await agent.run(sample_scene)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Shot) for s in result)
        assert result[0].scene_id == sample_scene.id
        assert result[1].dialogue == "There is a dragon..."

    @pytest.mark.asyncio
    async def test_shot_type_parsing(
        self, mock_llm: AsyncMock, sample_scene: Scene
    ) -> None:
        """Test various shot type string formats are parsed correctly."""
        mock_llm.complete.return_value = """{
            "shots": [
                {"id": 1, "description": "test", "duration": 5, "shot_type": "close-up"},
                {"id": 2, "description": "test", "duration": 5, "shot_type": "WIDE"},
                {"id": 3, "description": "test", "duration": 5, "shot_type": null}
            ]
        }"""

        agent = ShotPlannerAgent(llm_client=mock_llm, max_retries=1)
        result = await agent.run(sample_scene)

        from core.models import ShotType
        assert result[0].shot_type == ShotType.CLOSE_UP
        assert result[1].shot_type == ShotType.WIDE
        assert result[2].shot_type is None
