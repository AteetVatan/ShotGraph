"""LLM client implementations."""

import json
import logging
from typing import TYPE_CHECKING

import httpx

from core.exceptions import ShotGraphError

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class LLMClientError(ShotGraphError):
    """Error during LLM API call."""

    pass


class TogetherLLMClient:
    """LLM client for Together.ai API."""

    def __init__(self, settings: "Settings"):
        """Initialize the Together.ai client.

        Args:
            settings: Application settings with API configuration.
        """
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model
        self._base_url = settings.llm_together_base_url
        self._timeout = settings.llm_together_timeout
        self._max_tokens = settings.llm_max_tokens

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Send a completion request to Together.ai.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature.

        Returns:
            The LLM response text.

        Raises:
            LLMClientError: If the API call fails.
        """
        if not self._api_key:
            raise LLMClientError("LLM API key not configured")

        logger.debug("Sending request to Together.ai model: %s", self._model)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": self._max_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug("Received response (%d chars)", len(content))
                return content

        except httpx.HTTPStatusError as e:
            raise LLMClientError(
                f"Together.ai API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise LLMClientError(f"Together.ai request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise LLMClientError(f"Unexpected response format: {e}") from e


class GroqLLMClient:
    """LLM client for Groq API."""

    def __init__(self, settings: "Settings"):
        """Initialize the Groq client.

        Args:
            settings: Application settings with API configuration.
        """
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model
        self._base_url = settings.llm_groq_base_url
        self._timeout = settings.llm_groq_timeout
        self._max_tokens = settings.llm_max_tokens

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Send a completion request to Groq.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature.

        Returns:
            The LLM response text.

        Raises:
            LLMClientError: If the API call fails.
        """
        if not self._api_key:
            raise LLMClientError("LLM API key not configured")

        logger.debug("Sending request to Groq model: %s", self._model)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": self._max_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug("Received response (%d chars)", len(content))
                return content

        except httpx.HTTPStatusError as e:
            raise LLMClientError(
                f"Groq API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise LLMClientError(f"Groq request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise LLMClientError(f"Unexpected response format: {e}") from e


class MockLLMClient:
    """Mock LLM client for testing and debug mode."""

    def __init__(self, settings: "Settings | None" = None):
        """Initialize the mock client.

        Args:
            settings: Optional settings (ignored in mock).
        """
        self._response_count = 0

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Return a mock response based on the prompt context.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature (ignored).

        Returns:
            A mock JSON response appropriate for the agent type.
        """
        self._response_count += 1
        logger.debug("MockLLMClient generating response #%d", self._response_count)

        # Detect agent type from system prompt and return appropriate mock
        if "split" in system_prompt.lower() and "scene" in system_prompt.lower():
            return self._mock_scene_response()
        elif "shot" in system_prompt.lower() or "director" in system_prompt.lower():
            return self._mock_shot_response()
        else:
            return '{"message": "Mock response"}'

    def _mock_scene_response(self) -> str:
        """Generate mock scene breakdown response."""
        return json.dumps({
            "scenes": [
                {
                    "id": 1,
                    "summary": "Introduction - Hero in the village",
                    "text": "Once upon a time in a small village...",
                },
                {
                    "id": 2,
                    "summary": "The Call to Adventure - Meeting the wizard",
                    "text": "One morning, the hero met an old wizard...",
                },
                {
                    "id": 3,
                    "summary": "The Journey Begins - Setting off",
                    "text": "The hero decided to embark on the quest...",
                },
            ]
        })

    def _mock_shot_response(self) -> str:
        """Generate mock shot planning response."""
        return json.dumps({
            "shots": [
                {
                    "id": 1,
                    "description": "Wide establishing shot of a medieval village at dawn, "
                    "golden sunlight streaming through thatched roofs",
                    "duration": 5,
                    "shot_type": "establishing",
                    "dialogue": None,
                },
                {
                    "id": 2,
                    "description": "Medium shot of the main character walking through the village, "
                    "greeting villagers",
                    "duration": 6,
                    "shot_type": "medium",
                    "dialogue": None,
                },
                {
                    "id": 3,
                    "description": "Close-up of a mysterious figure in robes approaching",
                    "duration": 4,
                    "shot_type": "close_up",
                    "dialogue": "Greetings, young traveler.",
                },
            ]
        })
