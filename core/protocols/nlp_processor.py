"""Protocol for NLP processing interface."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from core.models import ProcessedStory, StoryEntities


class INLPProcessor(Protocol):
    """Interface for NLP preprocessing services."""

    async def preprocess(
        self,
        text: str,
        *,
        max_tokens: int = 4096,
        summarize_if_long: bool = True,
        skip_summarization_threshold: int = 2000,
    ) -> "ProcessedStory":
        """Preprocess story text for the pipeline.

        Handles chunking, summarization, and entity extraction.

        Args:
            text: The raw story text.
            max_tokens: Maximum tokens per chunk for LLM context.
            summarize_if_long: Whether to generate summary for long texts.
            skip_summarization_threshold: Skip summarization if tokens < threshold.

        Returns:
            ProcessedStory with chunks, summary, and entities.
        """
        ...

    def extract_entities(self, text: str) -> "StoryEntities":
        """Extract named entities from story text.

        Uses NER to identify characters, locations, and themes.

        Args:
            text: The story text to analyze.

        Returns:
            StoryEntities with extracted information.
        """
        ...

    async def summarize(
        self,
        text: str,
        *,
        max_length: int = 500,
    ) -> str:
        """Summarize text to reduce context size.

        Args:
            text: The text to summarize.
            max_length: Maximum summary length in characters.

        Returns:
            Condensed summary of the text.
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate the token count for text.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated token count.
        """
        ...

    def chunk_text(
        self,
        text: str,
        *,
        max_tokens: int = 4096,
    ) -> list[str]:
        """Split text into chunks respecting token limits.

        Args:
            text: The text to chunk.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of text chunks.
        """
        ...
