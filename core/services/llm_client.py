"""LLM client implementations."""

import json
import logging
from typing import TYPE_CHECKING, Any

from together import Together

from core.constants import FieldNames
from core.exceptions import ShotGraphError

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class LLMClientError(ShotGraphError):
    """Error during LLM API call."""

    pass


class TogetherLLMClient:
    """LLM client for Together.ai API using the Together SDK."""

    def __init__(self, settings: "Settings"):
        """Initialize the Together.ai client.

        Args:
            settings: Application settings with API configuration.
        """
        if not settings.llm_api_key:
            raise LLMClientError("LLM API key not configured")

        self._client = Together(api_key=settings.llm_api_key)
        self._default_model = settings.llm_model
        self._timeout = settings.llm_together_timeout
        self._max_tokens = settings.llm_max_tokens

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        safety_model: str | None = None,
    ) -> str:
        """Send a completion request to Together.ai.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature.
            model: Optional model override (uses default if None).
            response_format: Optional structured output format.
            safety_model: Optional safety/moderation model.

        Returns:
            The LLM response text.

        Raises:
            LLMClientError: If the API call fails.
        """
        model_name = model or self._default_model
        logger.debug("Sending request to Together.ai model: %s", model_name)

        try:
            # Build request parameters
            request_params: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": self._max_tokens,
            }

            # Add optional parameters if provided
            if response_format:
                request_params["response_format"] = response_format

            if safety_model:
                request_params["safety_model"] = safety_model

            # Use Together SDK
            response = self._client.chat.completions.create(**request_params)

            content = response.choices[0].message.content
            logger.debug("Received response (%d chars)", len(content))
            return content

        except Exception as e:
            # Extract more detailed error information from Together SDK error objects
            error_msg = str(e)
            
            # Check for Together SDK error response structure
            if hasattr(e, "api_response"):
                try:
                    api_response = e.api_response
                    error_type = getattr(api_response, "type_", None) or "unknown_error"
                    error_message = getattr(api_response, "message", error_msg) or error_msg
                    error_param = getattr(api_response, "param", None)
                    error_code = getattr(api_response, "code", None)
                    
                    if error_param:
                        error_msg = f"{error_type}: {error_message} (param: {error_param})"
                    elif error_code:
                        error_msg = f"{error_type}: {error_message} (code: {error_code})"
                    else:
                        error_msg = f"{error_type}: {error_message}"
                except AttributeError:
                    pass
            
            # Provide helpful context for common errors
            if "invalid_request_error" in error_msg.lower() or "validation" in error_msg.lower():
                model_name = request_params.get("model", "")
                if "Llama-Guard" in str(model_name):
                    error_msg = (
                        f"{error_msg}. Note: Llama-Guard models are not available as "
                        "chat completion models. Use a regular chat model instead."
                    )
            
            raise LLMClientError(f"Together.ai API error: {error_msg}") from e


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
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        safety_model: str | None = None,
    ) -> str:
        """Return a mock response based on the prompt context.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature (ignored).
            model: Optional model override (ignored in mock).
            response_format: Optional structured output format (ignored in mock).
            safety_model: Optional safety/moderation model (ignored in mock).

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
            FieldNames.SCENES: [
                {
                    FieldNames.ID: 1,
                    FieldNames.SUMMARY: "Introduction - Hero in the village",
                    FieldNames.TEXT: "Once upon a time in a small village...",
                },
                {
                    FieldNames.ID: 2,
                    FieldNames.SUMMARY: "The Call to Adventure - Meeting the wizard",
                    FieldNames.TEXT: "One morning, the hero met an old wizard...",
                },
                {
                    FieldNames.ID: 3,
                    FieldNames.SUMMARY: "The Journey Begins - Setting off",
                    FieldNames.TEXT: "The hero decided to embark on the quest...",
                },
            ]
        })

    def _mock_shot_response(self) -> str:
        """Generate mock shot planning response."""
        return json.dumps({
            FieldNames.SHOTS: [
                {
                    FieldNames.ID: 1,
                    FieldNames.DESCRIPTION: "Wide establishing shot of a medieval village at dawn, "
                    "golden sunlight streaming through thatched roofs",
                    FieldNames.DURATION: 5,
                    FieldNames.SHOT_TYPE: "establishing",
                    FieldNames.DIALOGUE: None,
                },
                {
                    FieldNames.ID: 2,
                    FieldNames.DESCRIPTION: "Medium shot of the main character walking through the village, "
                    "greeting villagers",
                    FieldNames.DURATION: 6,
                    FieldNames.SHOT_TYPE: "medium",
                    FieldNames.DIALOGUE: None,
                },
                {
                    FieldNames.ID: 3,
                    FieldNames.DESCRIPTION: "Close-up of a mysterious figure in robes approaching",
                    FieldNames.DURATION: 4,
                    FieldNames.SHOT_TYPE: "close_up",
                    FieldNames.DIALOGUE: "Greetings, young traveler.",
                },
            ]
        })
