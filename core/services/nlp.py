"""NLP preprocessing service for story text."""

import logging
import re
from typing import TYPE_CHECKING

from core.models import ProcessedStory, StoryEntities

if TYPE_CHECKING:
    from core.services.model_router import ModelRouter

logger = logging.getLogger(__name__)

# Chapter/section markers for splitting
CHAPTER_PATTERNS = [
    r"^Chapter\s+\d+",
    r"^CHAPTER\s+\d+",
    r"^Part\s+\d+",
    r"^PART\s+\d+",
    r"^Section\s+\d+",
    r"^Act\s+\d+",
    r"^\*\*\*",
    r"^---",
    r"^___",
]


class StoryPreprocessor:
    """NLP preprocessing service for story text.

    Handles:
    - Token estimation
    - Text chunking by chapters/paragraphs
    - Entity extraction using spaCy
    - Summarization via LLM
    """

    def __init__(
        self,
        *,
        model_router: "ModelRouter | None" = None,
        use_spacy: bool = True,
    ):
        """Initialize the preprocessor.

        Args:
            model_router: Optional model router for cost-optimized summarization.
            use_spacy: Whether to use spaCy for NER (fallback to regex if False).
        """
        self._router = model_router
        self._use_spacy = use_spacy
        self._nlp = None
        self._tokenizer = None

    def _load_spacy(self):
        """Lazy load spaCy model."""
        if self._nlp is not None:
            return

        if not self._use_spacy:
            return

        try:
            import spacy

            # Try to load the model, download if not present
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("Downloading spaCy model en_core_web_sm...")
                from spacy.cli import download

                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")

            logger.info("spaCy model loaded successfully")
        except ImportError:
            logger.warning("spaCy not installed, using regex-based entity extraction")
            self._use_spacy = False

    def _load_tokenizer(self):
        """Lazy load tiktoken tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            import tiktoken

            # Use cl100k_base encoding (GPT-4, Claude, etc.)
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.debug("tiktoken loaded successfully")
        except ImportError:
            logger.warning("tiktoken not installed, using word-based estimation")

    def estimate_tokens(self, text: str) -> int:
        """Estimate the token count for text.

        Uses tiktoken if available, otherwise estimates based on words.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated token count.
        """
        self._load_tokenizer()

        if self._tokenizer:
            return len(self._tokenizer.encode(text))

        # Fallback: ~1.3 tokens per word on average
        words = len(text.split())
        return int(words * 1.3)

    def chunk_text(
        self,
        text: str,
        *,
        max_tokens: int = 4096,
        overlap_tokens: int = 100,
    ) -> list[str]:
        """Split text into chunks respecting token limits.

        First tries to split by chapter markers, then by paragraphs.

        Args:
            text: The text to chunk.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Token overlap between chunks for context.

        Returns:
            List of text chunks.
        """
        # If text fits in one chunk, return as-is
        if self.estimate_tokens(text) <= max_tokens:
            return [text]

        logger.info("Text exceeds token limit, chunking...")

        # Try to split by chapters first
        chunks = self._split_by_chapters(text)

        # If no chapter splits, split by paragraphs
        if len(chunks) == 1:
            chunks = self._split_by_paragraphs(text, max_tokens, overlap_tokens)

        # Verify and re-split any oversized chunks
        final_chunks = []
        for chunk in chunks:
            if self.estimate_tokens(chunk) > max_tokens:
                # Force split by sentences
                sub_chunks = self._split_by_sentences(chunk, max_tokens, overlap_tokens)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        logger.info("Split text into %d chunks", len(final_chunks))
        return final_chunks

    def _split_by_chapters(self, text: str) -> list[str]:
        """Split text by chapter/section markers.

        Args:
            text: The text to split.

        Returns:
            List of chapter chunks.
        """
        # Build combined pattern
        combined_pattern = "|".join(f"({p})" for p in CHAPTER_PATTERNS)

        # Split by chapter markers
        parts = re.split(f"({combined_pattern})", text, flags=re.MULTILINE)

        # Recombine: keep markers with their content
        chunks = []
        current_chunk = ""

        for part in parts:
            if part is None:
                continue
            part = part.strip()
            if not part:
                continue

            # Check if this is a chapter marker
            is_marker = any(re.match(p, part, re.MULTILINE) for p in CHAPTER_PATTERNS)

            if is_marker and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = part + "\n"
            else:
                current_chunk += part + "\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if len(chunks) > 1 else [text]

    def _split_by_paragraphs(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split text by paragraphs respecting token limits.

        Args:
            text: The text to split.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Token overlap between chunks.

        Returns:
            List of paragraph-based chunks.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.estimate_tokens(para)

            if current_tokens + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from end of previous chunk
                    overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                    current_chunk = overlap_text + "\n\n" + para
                    current_tokens = self.estimate_tokens(current_chunk)
                else:
                    current_chunk = para
                    current_tokens = para_tokens
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_sentences(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split text by sentences as last resort.

        Args:
            text: The text to split.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Token overlap between chunks.

        Returns:
            List of sentence-based chunks.
        """
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_tokens = self.estimate_tokens(sentence)

            if current_tokens + sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sent_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sent_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens worth of text for overlap.

        Args:
            text: The source text.
            overlap_tokens: Target number of tokens.

        Returns:
            Overlap text from end of source.
        """
        words = text.split()
        # Approximate: take last (overlap_tokens / 1.3) words
        num_words = max(1, int(overlap_tokens / 1.3))
        overlap_words = words[-num_words:]
        return " ".join(overlap_words)

    def extract_entities(self, text: str) -> StoryEntities:
        """Extract named entities from story text.

        Uses spaCy NER if available, falls back to regex patterns.

        Args:
            text: The story text to analyze.

        Returns:
            StoryEntities with extracted information.
        """
        self._load_spacy()

        if self._nlp:
            return self._extract_entities_spacy(text)
        return self._extract_entities_regex(text)

    def _extract_entities_spacy(self, text: str) -> StoryEntities:
        """Extract entities using spaCy NER.

        Args:
            text: The text to analyze.

        Returns:
            Extracted entities.
        """
        # AB - Make this dynamic and configurableProcess text (limit to first 100k chars for performance)
        doc = self._nlp(text[:100000])

        characters = set()
        locations = set()
        organizations = set()

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                characters.add(ent.text)
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                locations.add(ent.text)
            elif ent.label_ == "ORG":
                organizations.add(ent.text)

        # Extract themes from noun chunks (simplified)
        themes = set()
        for chunk in doc.noun_chunks:
            # Look for abstract nouns that might be themes
            if chunk.root.pos_ == "NOUN" and len(chunk.text) > 3:
                if chunk.root.dep_ in ("nsubj", "dobj", "pobj"):
                    themes.add(chunk.text.lower())

        # AB check - Limit themes to most common
        themes = list(themes)[:10]

        logger.info(
            "Extracted entities - Characters: %d, Locations: %d, Orgs: %d",
            len(characters),
            len(locations),
            len(organizations),
        )
        # AB - Check
        return StoryEntities(
            characters=sorted(characters)[:20],  # Limit to top 20
            locations=sorted(locations)[:15],
            themes=themes,
            organizations=sorted(organizations)[:10],
        )

    def _extract_entities_regex(self, text: str) -> StoryEntities:
        """Extract entities using regex patterns (fallback).

        Args:
            text: The text to analyze.

        Returns:
            Extracted entities (basic extraction).
        """
        # Find capitalized names (very basic)
        name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
        potential_names = re.findall(name_pattern, text)

        # Filter common non-names
        stop_words = {
            "The",
            "This",
            "That",
            "There",
            "When",
            "Where",
            "What",
            "How",
            "Why",
            "But",
            "And",
            "Then",
            "Now",
            "Here",
            "Just",
            "Once",
            "Upon",
            "Time",
        }

        characters = []
        for name in potential_names:
            if name not in stop_words and len(name) > 2:
                characters.append(name)

        # Count frequency and take most common
        from collections import Counter

        name_counts = Counter(characters)
        top_characters = [name for name, _ in name_counts.most_common(15)]

        return StoryEntities(
            characters=top_characters,
            locations=[],
            themes=[],
            organizations=[],
        )

    async def summarize(
        self,
        text: str,
        *,
        max_length: int = 500,
    ) -> str:
        """Summarize text using LLM.

        Args:
            text: The text to summarize.
            max_length: Maximum summary length in characters.

        Returns:
            Condensed summary of the text.
        """
        if not self._llm:
            # Fallback: return first N characters
            logger.warning("No LLM client for summarization, using truncation")
            return text[:max_length] + "..." if len(text) > max_length else text

        system_prompt = """You are a skilled summarizer. Create a concise summary of the story that:
1. Captures the main plot points
2. Identifies key characters
3. Preserves the narrative arc
4. Is suitable for guiding scene breakdown

Output ONLY the summary, no additional commentary."""

        user_prompt = f"""Summarize this story in approximately {max_length} characters:

{text[:20000]}"""  # Limit input size

        try:
            summary = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
            )
            return summary.strip()[:max_length]
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return text[:max_length] + "..."

    async def preprocess(
        self,
        text: str,
        *,
        max_tokens: int = 4096,
        summarize_if_long: bool = True,
        skip_summarization_threshold: int = 2000,
    ) -> ProcessedStory:
        """Preprocess story text for the pipeline.

        Args:
            text: The raw story text.
            max_tokens: Maximum tokens per chunk for LLM context.
            summarize_if_long: Whether to generate summary for long texts.
            skip_summarization_threshold: Skip summarization if tokens < threshold.

        Returns:
            ProcessedStory with chunks, summary, and entities.
        """
        logger.info("Preprocessing story (length: %d chars)", len(text))

        # Estimate tokens
        token_count = self.estimate_tokens(text)
        logger.info("Estimated tokens: %d", token_count)

        # Extract entities
        entities = self.extract_entities(text)

        # Chunk if needed
        chunks = self.chunk_text(text, max_tokens=max_tokens)
        was_chunked = len(chunks) > 1

        # Cost optimization: Skip summarization for short stories
        summary = None
        if summarize_if_long and token_count > max_tokens * 2:
            # Check threshold before spending LLM cost
            if token_count >= skip_summarization_threshold:
                logger.info("Text is long, generating summary...")
                summary = await self.summarize(text, max_length=1000)
            else:
                logger.info(
                    "Skipping summarization (tokens=%d < threshold=%d, saves cost)",
                    token_count,
                    skip_summarization_threshold,
                )

        processed = ProcessedStory(
            original_text=text,
            chunks=chunks,
            summary=summary,
            entities=entities,
            token_count=token_count,
            was_chunked=was_chunked,
        )

        logger.info(
            "Preprocessing complete - Chunks: %d, Entities: %d chars, %d locs",
            len(chunks),
            len(entities.characters),
            len(entities.locations),
        )

        return processed


class MockNLPProcessor:
    """Mock NLP processor for testing/debug mode."""

    async def preprocess(
        self,
        text: str,
        *,
        max_tokens: int = 4096,
        summarize_if_long: bool = True,
        skip_summarization_threshold: int = 2000,
    ) -> ProcessedStory:
        """Return basic processed story without NLP.
        
        Args:
            text: The raw story text.
            max_tokens: Maximum tokens per chunk (ignored in mock).
            summarize_if_long: Whether to generate summary (ignored in mock).
            skip_summarization_threshold: Skip summarization threshold (ignored in mock).
        
        Returns:
            ProcessedStory with basic processing.
        """
        return ProcessedStory(
            original_text=text,
            chunks=[text],
            summary=None,
            entities=StoryEntities(),
            token_count=len(text.split()) * 2,
            was_chunked=False,
        )

    def extract_entities(self, text: str) -> StoryEntities:
        """Return empty entities."""
        return StoryEntities()

    async def summarize(self, text: str, *, max_length: int = 500) -> str:
        """Return truncated text."""
        return text[:max_length]

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens from word count."""
        return len(text.split()) * 2

    def chunk_text(self, text: str, *, max_tokens: int = 4096) -> list[str]:
        """Return text as single chunk."""
        return [text]
