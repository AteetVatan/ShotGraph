"""JSON repair agent for validating and fixing malformed JSON responses."""

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from core.exceptions import LLMParseError
from core.protocols.llm_client import ILLMClient
from core.schemas.json_schemas import get_scene_list_schema, get_shot_list_schema

from .base import BaseAgent

if TYPE_CHECKING:
    from core.services.model_router import ModelRouter

logger = logging.getLogger(__name__)


class JSONRepairInput(BaseModel):
    """Input model for JSON repair."""

    malformed_json: str = Field(..., description="Malformed JSON string to repair")
    schema_type: str = Field(default="scene_list", description="Type of schema to use")


class JSONRepairAgent(BaseAgent[JSONRepairInput, dict[str, Any]]):
    """Validates and repairs malformed JSON using structured outputs."""

    def __init__(
        self,
        *,
        model_router: "ModelRouter",
        max_retries: int = 1,
    ):
        """Initialize the JSON repair agent.

        Args:
            model_router: Model router for Step D calls.
            max_retries: Maximum retry attempts.
        """
        super().__init__(max_retries=max_retries)
        self._router = model_router

    async def _execute(self, input_data: JSONRepairInput) -> dict[str, Any]:
        """Repair JSON using Together's structured outputs.

        Args:
            input_data: Input containing malformed JSON and schema type.

        Returns:
            Repaired dictionary.

        Raises:
            LLMParseError: If repair fails.
        """
        # Get appropriate schema
        if input_data.schema_type == "scene_list":
            json_schema = get_scene_list_schema()
        elif input_data.schema_type == "shot_list":
            json_schema = get_shot_list_schema()
        else:
            raise ValueError(f"Unknown schema type: {input_data.schema_type}")

        malformed_json = input_data.malformed_json

        system_prompt = """You are a JSON repair specialist. Your task is to fix any syntax errors, missing quotes, brackets, or invalid structure in the provided JSON. Return ONLY valid JSON that matches the required schema. Do not add any explanation or commentary."""

        user_prompt = f"""Repair this malformed JSON to match the schema. Return only the corrected JSON, no other text:

{malformed_json[:5000]}"""  # Limit input size

        try:
            response = await self._router.call_stage_d(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                temperature=0.2,
            )

            # Structured output should be valid JSON
            # Try to extract JSON if wrapped in markdown
            json_str = self._extract_json(response)
            repaired = json.loads(json_str)

            logger.debug("Successfully repaired JSON")
            return repaired

        except json.JSONDecodeError as e:
            raise LLMParseError(
                f"JSON repair failed: invalid JSON in response: {e}",
                raw_response=response,
            ) from e
        except Exception as e:
            raise LLMParseError(
                f"JSON repair failed: {e}",
                raw_response=input_data.malformed_json,
            ) from e

    def _extract_json(self, response: str) -> str:
        """Extract JSON from response, handling markdown code blocks.

        Args:
            response: The raw response.

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
