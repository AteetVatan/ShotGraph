"""Shot planner agent for breaking scenes into shots."""

import json
import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from config.prompt_loader import (
    PROMPT_SHOT_PLANNING,
    PROMPT_STYLE_GUIDE,
    load_prompt,
)
from core.exceptions import LLMParseError
from core.models import Scene, Shot, ShotType
from core.protocols.llm_client import ILLMClient
from core.services.toon import TOONCodec

from .base import BaseAgent

if TYPE_CHECKING:
    from config.settings import Settings
    from core.services.style_context import StyleContextManager

logger = logging.getLogger(__name__)


class ShotPlannerAgent(BaseAgent[Scene, list[Shot]]):
    """Agent that plans shots for a scene using an LLM."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient,
        max_retries: int = 2,
        system_prompt: str | None = None,
        style_context_manager: "StyleContextManager | None" = None,
        settings: "Settings | None" = None,
    ):
        """Initialize the shot planner agent.

        Args:
            llm_client: LLM client for text generation.
            max_retries: Maximum retry attempts.
            system_prompt: Optional custom system prompt.
            style_context_manager: Optional style context for consistency.
            settings: Optional settings for TOON format configuration.
        """
        super().__init__(max_retries=max_retries)
        self._llm = llm_client
        self._style_ctx = style_context_manager
        self._use_toon = settings.use_toon_format if settings else False
        self._toon_codec = TOONCodec() if self._use_toon else None
        
        if system_prompt is None:
            try:
                shot_prompt = load_prompt(
                    PROMPT_SHOT_PLANNING,
                    use_toon=self._use_toon,
                )
                try:
                    style_guide = load_prompt(PROMPT_STYLE_GUIDE)
                    self._system_prompt = f"{shot_prompt}\n\n{style_guide}"
                except (FileNotFoundError, IOError) as e:
                    logger.warning(
                        "Style guide not found, using shot planning prompt only: %s",
                        e,
                    )
                    self._system_prompt = shot_prompt
            except (FileNotFoundError, IOError) as e:
                logger.error("Failed to load shot planning prompt: %s", e)
                raise
        else:
            self._system_prompt = system_prompt

    def set_style_context_manager(self, manager: "StyleContextManager") -> None:
        """Set the style context manager.

        Args:
            manager: The style context manager to use.
        """
        self._style_ctx = manager

    async def _execute(self, scene: Scene) -> list[Shot]:
        """Plan shots for the given scene.

        Args:
            scene: The scene to plan shots for.

        Returns:
            A list of Shot objects for the scene.

        Raises:
            LLMParseError: If the LLM response cannot be parsed.
        """
        self.logger.info(
            "Planning shots for scene %d: %s (toon=%s)",
            scene.id,
            scene.summary[:50],
            self._use_toon,
        )

        # Build user prompt with style context if available
        user_prompt = self._build_user_prompt(scene)

        response = await self._llm.complete(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,  # Slightly higher for creative shot descriptions
        )

        self.logger.debug("LLM response: %s", response[:200])

        try:
            # Parse response based on format
            data = self._parse_response(response)

            shots = []
            for idx, shot_data in enumerate(data.get("shots", [])):
                # Map shot_type string to enum
                shot_type_str = shot_data.get("shot_type")
                shot_type = self._parse_shot_type(shot_type_str)

                # Handle null dialogue from TOON format
                dialogue = shot_data.get("dialogue")
                if dialogue in ("null", "None", None):
                    dialogue = None

                shot = Shot(
                    id=shot_data["id"],
                    scene_id=scene.id,
                    description=shot_data["description"],
                    duration_seconds=float(shot_data.get("duration", 5.0)),
                    shot_type=shot_type,
                    dialogue=dialogue,
                    subtitle_text=dialogue,  # Use dialogue as subtitle
                )
                shots.append(shot)

                # Record shot in style context for continuity
                if self._style_ctx:
                    self._style_ctx.record_shot(
                        scene_id=scene.id,
                        shot_index=idx,
                        summary=shot_data["description"][:100],
                        visual_style=shot_data.get("visual_style", ""),
                    )

            if not shots:
                raise LLMParseError(
                    f"No shots found for scene {scene.id}",
                    raw_response=response,
                )

            self.logger.info("Successfully planned %d shots for scene %d", len(shots), scene.id)
            return shots

        except json.JSONDecodeError as e:
            raise LLMParseError(
                f"Invalid JSON in shot planning response: {e}",
                raw_response=response,
            ) from e
        except (KeyError, ValidationError) as e:
            raise LLMParseError(
                f"Invalid shot structure in response: {e}",
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

    def _build_user_prompt(self, scene: Scene) -> str:
        """Build user prompt with style context.

        Args:
            scene: The scene to build prompt for.

        Returns:
            Formatted prompt string.
        """
        parts = []

        # Add style context if available
        if self._style_ctx:
            # Build context for first shot (will be updated per-shot if needed)
            ctx = self._style_ctx.build_context_for_shot(scene, 0)
            context_text = self._style_ctx.format_for_prompt(ctx)
            if context_text:
                parts.append(context_text)
                parts.append("")

        # Add scene information
        parts.append(f"Scene {scene.id}: {scene.summary}")
        parts.append("")
        parts.append("Full scene text:")
        parts.append(scene.text)
        parts.append("")
        parts.append("Create a sequence of cinematic shots for this scene.")

        if self._style_ctx:
            parts.append("")
            parts.append("IMPORTANT: Maintain visual consistency with:")
            parts.append("- Use consistent character descriptions across all shots")
            parts.append("- Keep the established setting and mood")
            parts.append("- Ensure continuity with previous shots")

        return "\n".join(parts)

    def _parse_shot_type(self, shot_type_str: str | None) -> ShotType | None:
        """Parse shot type string to enum.

        Args:
            shot_type_str: The shot type string from LLM.

        Returns:
            The corresponding ShotType enum or None.
        """
        if not shot_type_str:
            return None

        mapping = {
            "wide": ShotType.WIDE,
            "medium": ShotType.MEDIUM,
            "close_up": ShotType.CLOSE_UP,
            "close-up": ShotType.CLOSE_UP,
            "closeup": ShotType.CLOSE_UP,
            "establishing": ShotType.ESTABLISHING,
        }

        return mapping.get(shot_type_str.lower())

    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response.

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
            depth = 0
            for i, char in enumerate(response[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return response[start : i + 1]

        return response
