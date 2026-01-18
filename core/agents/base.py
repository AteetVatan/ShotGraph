"""Base agent class with retry logic."""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from core.exceptions import RetryExhaustedError

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput")


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """Abstract base class for all pipeline agents.

    Provides common functionality including:
    - Structured logging
    - Retry logic with configurable attempts
    - Consistent input/output handling

    Subclasses must implement the `_execute` method with their specific logic.
    """

    def __init__(self, *, max_retries: int = 2):
        """Initialize the agent.

        Args:
            max_retries: Maximum number of retry attempts (0 = no retries).
        """
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def _execute(self, input_data: TInput) -> TOutput:
        """Core execution logic - must be implemented by subclasses.

        Args:
            input_data: The input data for this agent.

        Returns:
            The processed output.

        Raises:
            Any exception that should trigger a retry.
        """
        ...

    async def run(self, input_data: TInput) -> TOutput:
        """Execute the agent with retry logic.

        Args:
            input_data: The input data to process.

        Returns:
            The processed output.

        Raises:
            RetryExhaustedError: If all retry attempts fail.
        """
        last_error: Exception | None = None
        total_attempts = self.max_retries + 1

        for attempt in range(total_attempts):
            try:
                self.logger.info(
                    "Executing attempt %d/%d",
                    attempt + 1,
                    total_attempts,
                )
                result = await self._execute(input_data)
                self.logger.info("Execution successful on attempt %d", attempt + 1)
                return result
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt + 1,
                    total_attempts,
                    str(e),
                )
                if attempt < self.max_retries:
                    self.logger.info("Retrying...")

        raise RetryExhaustedError(
            f"All {total_attempts} attempts failed for {self.__class__.__name__}",
            attempts=total_attempts,
        ) from last_error
