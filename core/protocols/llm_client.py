"""Protocol for LLM client interface."""

from typing import Protocol


class ILLMClient(Protocol):
    """Interface for Language Model clients."""

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Send a prompt to the LLM and return the response text.

        Args:
            system_prompt: The system instruction for the LLM.
            user_prompt: The user message/query.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            The LLM's response text.

        Raises:
            ShotGraphError: If the LLM call fails.
        """
        ...
