"""Scene splitter agent for breaking stories into scenes."""

import json
import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from config.prompt_loader import PROMPT_SCENE_BREAKDOWN, load_prompt
from core.exceptions import LLMParseError
from core.models import ProcessedStory, Scene, SceneList, StoryInput
from core.protocols.llm_client import ILLMClient
from core.services.toon import TOONCodec

from .base import BaseAgent

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class SceneSplitterAgent(BaseAgent[StoryInput, SceneList]):
    """Agent that splits a story into scenes using an LLM."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient,
        max_retries: int = 2,
        system_prompt: str | None = None,
        settings: "Settings | None" = None,
    ):
        """Initialize the scene splitter agent.

        Args:
            llm_client: LLM client for text generation.
            max_retries: Maximum retry attempts.
            system_prompt: Optional custom system prompt.
            settings: Optional settings for TOON format configuration.
        """
        super().__init__(max_retries=max_retries)
        self._llm = llm_client
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

        response = await self._llm.complete(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # Lower temperature for more consistent output
        )

        self.logger.debug("LLM response: %s", response[:200])

        try:
            # Parse response based on format
            data = self._parse_response(response)

            # Validate and create SceneList
            scenes = []
            for scene_data in data.get("scenes", []):
                scene = Scene(
                    id=scene_data["id"],
                    summary=scene_data["summary"],
                    text=scene_data["text"],
                )
                scenes.append(scene)

            if not scenes:
                raise LLMParseError(
                    "No scenes found in LLM response",
                    raw_response=response,
                )

            scene_list = SceneList(scenes=scenes)
            self.logger.info("Successfully parsed %d scenes", len(scenes))
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
