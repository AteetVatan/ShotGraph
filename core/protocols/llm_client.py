"""Protocol for LLM client interface."""

from typing import Any, Protocol


class ILLMClient(Protocol):
    """Interface for Language Model clients."""

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
        """Send a prompt to the LLM and return the response text.

        Args:
            system_prompt: The system instruction for the LLM.
            user_prompt: The user message/query.
            temperature: Sampling temperature (0.0-1.0).
            model: Optional model override (uses default if None).
            response_format: Optional structured output format (e.g., json_schema).
            safety_model: Optional safety/moderation model.

        Returns:
            The LLM's response text.

        Raises:
            ShotGraphError: If the LLM call fails.
        """
        ...
