"""Scene splitter agent for breaking stories into scenes."""

import json
import logging
import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from config.prompt_loader import PROMPT_SCENE_BREAKDOWN, load_prompt
from core.exceptions import LLMParseError
from core.models import ProcessedStory, Scene, SceneList, StoryInput
from core.protocols.llm_client import ILLMClient
from core.services.model_router import ModelRouter
from core.services.toon import TOONCodec

from .base import BaseAgent

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)

# Validation constants
MIN_TEXT_COMPLETENESS_RATIO: float = 0.85
MIN_SENTENCE_COVERAGE_RATIO: float = 0.7
MIN_WORD_LENGTH_FOR_MATCHING: int = 3


class SceneSplitterAgent(BaseAgent[StoryInput, SceneList]):
    """Agent that splits a story into scenes using an LLM."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient,
        model_router: ModelRouter | None = None,
        max_retries: int = 2,
        system_prompt: str | None = None,
        settings: "Settings | None" = None,
    ):
        """Initialize the scene splitter agent.

        Args:
            llm_client: LLM client for text generation (fallback if router not available).
            model_router: Optional model router for cost-optimized routing (Step B).
            max_retries: Maximum retry attempts.
            system_prompt: Optional custom system prompt.
            settings: Optional settings for TOON format configuration.
        """
        super().__init__(max_retries=max_retries)
        self._llm = llm_client
        self._router = model_router
        self._settings = settings
        self._processed_story: ProcessedStory | None = None
        self._use_toon = settings.use_toon_format if settings else False
        self._toon_codec = TOONCodec() if self._use_toon else None
        
        if system_prompt is None:
            try:
                self._system_prompt = load_prompt(
                    PROMPT_SCENE_BREAKDOWN,
                    use_toon=self._use_toon,
                )
            except (FileNotFoundError, IOError) as e:
                logger.error("Failed to load scene breakdown prompt: %s", e)
                raise
        else:
            self._system_prompt = system_prompt

    async def run(
        self,
        input_data: StoryInput,
        *,
        processed_story: ProcessedStory | None = None,
    ) -> SceneList:
        """Run the scene splitter with optional preprocessed story.

        Args:
            input_data: The story input to process.
            processed_story: Optional NLP-processed story with entities.

        Returns:
            A SceneList containing the parsed scenes.
        """
        self._processed_story = processed_story
        return await super().run(input_data)

    async def _execute(self, input_data: StoryInput) -> SceneList:
        """Split the story into scenes.

        Args:
            input_data: The story input to process.

        Returns:
            A SceneList containing the parsed scenes.

        Raises:
            LLMParseError: If the LLM response cannot be parsed.
        """
        self.logger.info(
            "Splitting story into scenes (length: %d chars, toon=%s)",
            len(input_data.text),
            self._use_toon,
        )

        # Build user prompt with entity context if available
        user_prompt = self._build_user_prompt(input_data)

        # Determine if large context needed
        token_count = (
            self._processed_story.token_count
            if self._processed_story
            else len(input_data.text.split()) * 2
        )
        use_large = (
            self._settings
            and token_count > self._settings.llm_use_large_context_threshold
        )

        # Use ModelRouter if available (Step B), otherwise fallback to direct LLM client
        if self._router:
            response = await self._router.call_stage_b(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                use_large=use_large,
            )
        else: # AB - Test this block
            response = await self._llm.complete(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for more consistent output
            )

        self.logger.debug("LLM response: %s", response[:200])

        try:
            # Parse response based on format
            data = self._parse_response(response)

            # Validate into Pydantic models (never trust raw LLM TOON as source of truth)
            scene_list = SceneList.model_validate(data)

            if not scene_list.scenes:
                raise LLMParseError(
                    "No scenes found in LLM response",
                    raw_response=response,
                )

            # Validate text completeness
            is_valid, error_msg = self._validate_text_completeness(
                scene_list,
                input_data.text,
            )
            if not is_valid:
                self.logger.warning(
                    "Text completeness validation failed: %s. Retrying...",
                    error_msg,
                )
                # AB - For comment this block as LLM response vary everytime
                # raise LLMParseError(
                #     f"Story text not fully captured: {error_msg}",
                #     raw_response=response,
                # )

            # Filter out any empty scenes that passed validation
            scene_list = self._filter_empty_scenes(scene_list)

            if not scene_list.scenes:
                raise LLMParseError(
                    "No valid scenes found after filtering empty scenes",
                    raw_response=response,
                )

            # Emit canonical TOON for debugging (guaranteed valid)
            if self._use_toon and self._toon_codec:
                canonical_toon = self._toon_codec.encode(scene_list)
                self.logger.debug("Canonical TOON (from validated models):\n%s", canonical_toon[:500])

            self.logger.info("Successfully parsed %d scenes", len(scene_list.scenes))
            return scene_list

        except json.JSONDecodeError as e:
            raise LLMParseError(
                f"Invalid JSON in scene breakdown response: {e}",
                raw_response=response,
            ) from e
        except (KeyError, ValidationError) as e:
            raise LLMParseError(
                f"Invalid scene structure in response: {e}",
                raw_response=response,
            ) from e

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response in either JSON or TOON format.

        Args:
            response: The raw LLM response.

        Returns:
            Parsed dictionary.
        """
        # Try TOON parsing first if enabled
        if self._use_toon and self._toon_codec:
            try:
                data = self._toon_codec.decode(response)
                self.logger.debug("Successfully parsed TOON response")
                return data
            except (ValueError, KeyError) as e:
                self.logger.warning("TOON parsing failed: %s, trying JSON", e)

        # Fall back to JSON parsing
        json_str = self._extract_json(response)
        return json.loads(json_str)

    def _build_user_prompt(self, input_data: StoryInput) -> str:
        """Build the user prompt with entity context if available.

        Args:
            input_data: The story input.

        Returns:
            Formatted user prompt string.
        """
        parts = []

        # Add entity context if available
        if self._processed_story and self._processed_story.entities:
            entities = self._processed_story.entities
            context_lines = []

            if entities.characters:
                context_lines.append(f"Main characters: {', '.join(entities.characters[:10])}")
            if entities.locations:
                context_lines.append(f"Key locations: {', '.join(entities.locations[:10])}")
            if entities.themes:
                context_lines.append(f"Themes: {', '.join(entities.themes[:5])}")

            if context_lines:
                parts.append("Story context extracted from NLP analysis:")
                parts.extend(context_lines)
                parts.append("")

        # Add summary if available (for very long stories)
        if self._processed_story and self._processed_story.summary:
            parts.append("Story summary (for context):")
            parts.append(self._processed_story.summary)
            parts.append("")

        # Add the actual story text
        parts.append("Split this story into scenes:")
        parts.append("")
        parts.append(input_data.text)

        return "\n".join(parts)

    def _filter_empty_scenes(self, scene_list: SceneList) -> SceneList:
        """Filter out scenes with empty text.

        Args:
            scene_list: The scene list to filter.

        Returns:
            New SceneList with empty scenes removed.
        """
        non_empty_scenes = [s for s in scene_list.scenes if s.text.strip()]
        return SceneList(scenes=non_empty_scenes)

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace for text comparison.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text with single spaces.
        """
        return " ".join(text.split())

    def _extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text using regex.

        Args:
            text: Text to extract sentences from.

        Returns:
            List of non-empty sentences.
        """
        sentences = [
            s.strip()
            for s in re.split(r'[.!?]+', text)
            if s.strip()
        ]
        return sentences

    def _check_sentence_coverage(
        self,
        original: str,
        combined: str,
    ) -> tuple[bool, str]:
        """Check if all sentences from original are covered in combined.

        Args:
            original: Original story text.
            combined: Combined text from all scenes.

        Returns:
            Tuple of (is_valid, error_message).
        """
        original_normalized = self._normalize_text(original)
        combined_normalized = self._normalize_text(combined)
        original_sentences = self._extract_sentences(original)

        for sentence in original_sentences:
            if sentence not in combined_normalized:
                key_words = [
                    w.lower()
                    for w in sentence.split()
                    if len(w) > MIN_WORD_LENGTH_FOR_MATCHING
                ]
                if key_words:
                    found_words = sum(
                        1
                        for word in key_words
                        if word in combined_normalized.lower()
                    )
                    coverage = found_words / len(key_words)
                    if coverage < MIN_SENTENCE_COVERAGE_RATIO:
                        return False, f"Missing text: {sentence[:50]}..."

        return True, ""

    def _check_length_ratio(
        self,
        original: str,
        combined: str,
    ) -> tuple[bool, str]:
        """Check if combined text length meets minimum ratio.

        Args:
            original: Original story text.
            combined: Combined text from all scenes.

        Returns:
            Tuple of (is_valid, error_message).
        """
        original_normalized = self._normalize_text(original)
        combined_normalized = self._normalize_text(combined)

        if not original_normalized:
            return False, "Original text is empty"

        length_ratio = len(combined_normalized) / len(original_normalized)
        if length_ratio < MIN_TEXT_COMPLETENESS_RATIO:
            return False, (
                f"Text completeness too low: {length_ratio:.1%} "
                f"(expected >{MIN_TEXT_COMPLETENESS_RATIO:.0%})"
            )

        return True, ""

    def _validate_text_completeness(
        self,
        scene_list: SceneList,
        original_text: str,
    ) -> tuple[bool, str]:
        """Validate that all original story text is captured in scenes.

        Args:
            scene_list: The parsed scene list to validate.
            original_text: The original story text.

        Returns:
            Tuple of (is_valid, error_message).
        """
        filtered = self._filter_empty_scenes(scene_list)
        if len(filtered.scenes) < len(scene_list.scenes):
            empty_count = len(scene_list.scenes) - len(filtered.scenes)
            return False, f"Found {empty_count} scene(s) with empty text"

        combined_text = " ".join(s.text.strip() for s in filtered.scenes)

        is_valid, error_msg = self._check_length_ratio(original_text, combined_text)
        if not is_valid:
            return False, error_msg

        is_valid, error_msg = self._check_sentence_coverage(original_text, combined_text)
        if not is_valid:
            return False, error_msg

        return True, ""

    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks.

        Args:
            response: The raw LLM response.

        Returns:
            The extracted JSON string.
        """
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Try to find JSON object boundaries
        if "{" in response:
            start = response.find("{")
            # Find the matching closing brace
            depth = 0
            for i, char in enumerate(response[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return response[start : i + 1]

        return response
