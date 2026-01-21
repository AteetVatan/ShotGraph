"""Model router for cost-optimized per-stage LLM routing."""

import logging
from typing import TYPE_CHECKING, Any

from core.protocols.llm_client import ILLMClient

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)

# Model pricing per million tokens: (input_price, output_price) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Step A - Story compression (cheapest)
    "google/gemma-3n-E4B-it": (0.02, 0.04),
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": (0.06, 0.06),
    # Step B - Scene breakdown
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
    # Step C - Shot planning
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
    # Step D - JSON repair
    # Safety/moderation
    "meta-llama/Llama-Guard-4-12B": (0.18, 0.18),  # Estimated, no pricing in docs
    # Default/fallback
    "mistralai/Mistral-7B-Instruct-v0.3": (0.20, 0.20),  # Estimated default
}


class ModelRouter:
    """Routes LLM calls to cost-optimized models per pipeline stage."""

    def __init__(self, settings: "Settings", llm_client: ILLMClient):
        """Initialize the model router.

        Args:
            settings: Application settings with model configuration.
            llm_client: LLM client for making API calls.
        """
        self._settings = settings
        self._client = llm_client
        self._cost_tracking: list[dict[str, Any]] = []

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model call.

        Args:
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in dollars.
        """
        pricing = MODEL_PRICING.get(model, (0.0, 0.0))
        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    def log_call(
        self,
        stage: str,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Log a model call for cost tracking.

        Args:
            stage: Pipeline stage (A, B, C, D).
            model: Model used.
            input_tokens: Estimated input tokens (if available).
            output_tokens: Estimated output tokens (if available).
        """
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = self.estimate_cost(model, input_tokens, output_tokens)

        call_info = {
            "stage": stage,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": cost,
        }
        self._cost_tracking.append(call_info)

        logger.info(
            "Model call - Stage: %s, Model: %s, Input: %s, Output: %s, Cost: $%.6f",
            stage,
            model,
            input_tokens or "unknown",
            output_tokens or "unknown",
            cost or 0.0,
        )

    def get_cost_summary(self) -> dict[str, Any]:
        """Get summary of costs by stage.

        Returns:
            Dictionary with cost breakdown by stage and total.
        """
        total_cost = 0.0
        by_stage: dict[str, float] = {}

        for call in self._cost_tracking:
            if call["estimated_cost"] is not None:
                cost = call["estimated_cost"]
                total_cost += cost
                stage = call["stage"]
                by_stage[stage] = by_stage.get(stage, 0.0) + cost

        return {
            "total_cost": total_cost,
            "by_stage": by_stage,
            "call_count": len(self._cost_tracking),
        }

    async def call_stage_a(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
    ) -> str:
        """Step A: Story compression - use cheapest model with fallback.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature.

        Returns:
            The LLM response text.

        Raises:
            LLMClientError: If both primary and fallback models fail.
        """
        try:
            model = self._settings.llm_model_story_compress
            logger.debug("Step A: Using primary model %s", model)
            response = await self._client.complete(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            # Estimate tokens for cost tracking (rough estimate: ~4 chars per token)
            input_tokens = len(system_prompt + user_prompt) // 4
            output_tokens = len(response) // 4
            self.log_call("A", model, input_tokens, output_tokens)
            return response
        except Exception as e:
            logger.warning("Step A primary model failed, using fallback: %s", e)
            model = self._settings.llm_model_story_compress_fallback
            response = await self._client.complete(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            input_tokens = len(system_prompt + user_prompt) // 4
            output_tokens = len(response) // 4
            self.log_call("A", model, input_tokens, output_tokens)
            return response

    async def call_stage_b(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        use_large: bool = False,
    ) -> str:
        """Step B: Scene breakdown - default or large context.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature.
            use_large: Whether to use large context model.

        Returns:
            The LLM response text.

        Raises:
            LLMClientError: If the API call fails.
        """
        model = (
            self._settings.llm_model_scene_draft_large
            if use_large
            else self._settings.llm_model_scene_draft
        )
        logger.debug("Step B: Using model %s (large=%s)", model, use_large)
        response = await self._client.complete( # AB - Check why it is called two times
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        input_tokens = len(system_prompt + user_prompt) // 4
        output_tokens = len(response) // 4
        self.log_call("B", model, input_tokens, output_tokens)
        return response

    async def call_stage_c(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.5,
        use_fallback: bool = False,
    ) -> str:
        """Step C: Shot planning - default or fallback for hard cases.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            temperature: Sampling temperature.
            use_fallback: Whether to use fallback model.

        Returns:
            The LLM response text.

        Raises:
            LLMClientError: If the API call fails.
        """
        model = (
            self._settings.llm_model_shot_final_fallback
            if use_fallback
            else self._settings.llm_model_shot_final
        )
        logger.debug("Step C: Using model %s (fallback=%s)", model, use_fallback)
        response = await self._client.complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        input_tokens = len(system_prompt + user_prompt) // 4
        output_tokens = len(response) // 4
        self.log_call("C", model, input_tokens, output_tokens)
        return response

    async def call_stage_d(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Step D: JSON repair with structured outputs.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user message.
            json_schema: Optional JSON schema for structured outputs.
            temperature: Sampling temperature.

        Returns:
            The LLM response text (should be valid JSON if schema provided).

        Raises:
            LLMClientError: If the API call fails.
        """
        response_format = None
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        model = self._settings.llm_model_json_repair
        logger.debug("Step D: Using model %s (structured=%s)", model, json_schema is not None)
        response = await self._client.complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            response_format=response_format,
        )
        input_tokens = len(system_prompt + user_prompt) // 4
        output_tokens = len(response) // 4
        self.log_call("D", model, input_tokens, output_tokens)
        return response

    async def check_safety(self, text: str) -> bool:
        """Check if text passes safety/moderation.

        Args:
            text: The text to check.

        Returns:
            True if safe, False if unsafe.

        Note:
            Currently fails open (returns True) on errors. This could be
            made configurable in the future.
        """
        system_prompt = "You are a content moderation system. Analyze the content and return 'SAFE' if appropriate, 'UNSAFE' if inappropriate."
        user_prompt = f"Moderate this content: {text[:1000]}"

        try:
            # Use a small regular chat model for safety checks
            # Llama-Guard models are not available as chat completion models
            # Use the JSON repair model (small, fast, cost-effective) for moderation
            model = self._settings.llm_model_json_repair
            response = await self._client.complete(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            is_safe = "SAFE" in response.upper()
            logger.debug("Safety check result: %s", "SAFE" if is_safe else "UNSAFE")
            input_tokens = len(system_prompt + user_prompt) // 4
            output_tokens = len(response) // 4
            self.log_call("safety", model, input_tokens, output_tokens)
            return is_safe
        except Exception as e:
            logger.error("Safety check failed: %s", e)
            # Fail open for now (could be configurable)
            return True
