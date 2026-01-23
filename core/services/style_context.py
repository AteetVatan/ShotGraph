"""Style context management for visual consistency across shots."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.constants import FieldNames

if TYPE_CHECKING:
    from core.models import Scene, Shot, StoryEntities

logger = logging.getLogger(__name__)


@dataclass
class CharacterAppearance:
    """Visual description of a character."""

    name: str
    description: str = ""
    first_appearance_scene: int = 0
    visual_traits: list[str] = field(default_factory=list)


@dataclass
class StyleContext:
    """Style context for a shot, enabling visual consistency."""

    scene_id: int
    shot_index: int
    
    # Character information
    characters: dict[str, str] = field(default_factory=dict)  # name -> visual description
    
    # Scene setting
    setting: str = ""
    time_of_day: str = ""
    weather: str = ""
    mood: str = ""
    
    # Continuity
    previous_shot_summary: str | None = None
    previous_shot_visual_style: str | None = None
    
    # Generation hints
    seed: int | None = None
    style_keywords: list[str] = field(default_factory=list)


class StyleContextManager:
    """Manages style context for maintaining visual consistency across shots.

    Tracks character appearances, scene settings, and provides context
    for shot planning and video generation to ensure continuity.
    """

    def __init__(self, *, entities: "StoryEntities | None" = None):
        """Initialize the style context manager.

        Args:
            entities: Optional extracted story entities for initial context.
        """
        self._characters: dict[str, CharacterAppearance] = {}
        self._scene_settings: dict[int, dict] = {}
        self._shot_history: list[dict] = []
        self._base_seed: int = 42
        
        # Initialize from entities if provided
        if entities:
            self._init_from_entities(entities)

    def _init_from_entities(self, entities: "StoryEntities") -> None:
        """Initialize character tracking from extracted entities.

        Args:
            entities: Story entities with character names.
        """
        for char_name in entities.characters:
            self._characters[char_name] = CharacterAppearance(
                name=char_name,
                description="",  # Will be filled during shot planning
            )
        logger.info("Initialized %d characters from entities", len(self._characters))

    def register_character_appearance(
        self,
        name: str,
        description: str,
        scene_id: int,
    ) -> None:
        """Register or update a character's visual appearance.

        Args:
            name: Character name.
            description: Visual description of the character.
            scene_id: Scene where this description was established.
        """
        if name not in self._characters:
            self._characters[name] = CharacterAppearance(
                name=name,
                description=description,
                first_appearance_scene=scene_id,
            )
        elif not self._characters[name].description:
            # Update description if not yet set
            self._characters[name].description = description
            self._characters[name].first_appearance_scene = scene_id
        
        logger.debug("Registered character '%s': %s", name, description[:50])

    def register_scene_setting(
        self,
        scene_id: int,
        *,
        setting: str = "",
        time_of_day: str = "",
        weather: str = "",
        mood: str = "",
    ) -> None:
        """Register scene setting information.

        Args:
            scene_id: Scene identifier.
            setting: Location/setting description.
            time_of_day: Time of day (dawn, day, dusk, night).
            weather: Weather conditions.
            mood: Scene mood/atmosphere.
        """
        self._scene_settings[scene_id] = {
            FieldNames.SETTING: setting,
            FieldNames.TIME_OF_DAY: time_of_day,
            FieldNames.WEATHER: weather,
            FieldNames.MOOD: mood,
        }

    def record_shot(
        self,
        scene_id: int,
        shot_index: int,
        summary: str,
        visual_style: str = "",
    ) -> None:
        """Record a shot for continuity tracking.

        Args:
            scene_id: Scene containing the shot.
            shot_index: Index of the shot within the scene.
            summary: Brief summary of shot content.
            visual_style: Visual style description used.
        """
        self._shot_history.append({
            FieldNames.SCENE_ID: scene_id,
            "shot_index": shot_index,
            FieldNames.SUMMARY: summary,
            FieldNames.VISUAL_STYLE: visual_style,
        })

    def build_context_for_shot(
        self,
        scene: "Scene",
        shot_index: int,
    ) -> StyleContext:
        """Build style context for a specific shot.

        Args:
            scene: The scene containing the shot.
            shot_index: Index of the shot within the scene.

        Returns:
            StyleContext with all relevant information.
        """
        scene_id = scene.id

        # Get scene settings
        scene_setting = self._scene_settings.get(scene_id, {})

        # Get previous shot info for continuity
        previous_summary = None
        previous_style = None
        
        if self._shot_history:
            last_shot = self._shot_history[-1]
            previous_summary = last_shot.get(FieldNames.SUMMARY)
            previous_style = last_shot.get(FieldNames.VISUAL_STYLE)

        # Build character context (filter to characters likely in this scene)
        char_context = {}
        scene_text_lower = scene.text.lower()
        for name, appearance in self._characters.items():
            if name.lower() in scene_text_lower and appearance.description:
                char_context[name] = appearance.description

        # Generate consistent seed for this shot
        shot_seed = self._base_seed + (scene_id * 100) + shot_index

        # Build style keywords from scene analysis
        style_keywords = self._extract_style_keywords(scene.text, scene.summary)

        context = StyleContext(
            scene_id=scene_id,
            shot_index=shot_index,
            characters=char_context,
            setting=scene_setting.get(FieldNames.SETTING, ""),
            time_of_day=scene_setting.get(FieldNames.TIME_OF_DAY, ""),
            weather=scene_setting.get(FieldNames.WEATHER, ""),
            mood=scene_setting.get(FieldNames.MOOD, ""),
            previous_shot_summary=previous_summary,
            previous_shot_visual_style=previous_style,
            seed=shot_seed,
            style_keywords=style_keywords,
        )

        logger.debug(
            "Built context for scene %d, shot %d: %d characters, seed=%d",
            scene_id,
            shot_index,
            len(char_context),
            shot_seed,
        )

        return context

    def _extract_style_keywords(self, text: str, summary: str) -> list[str]:
        """Extract style keywords from scene text.

        Args:
            text: Full scene text.
            summary: Scene summary.

        Returns:
            List of style keywords.
        """
        keywords = []
        combined = (text + " " + summary).lower()

        # Time-related
        if any(word in combined for word in ["dawn", "sunrise", "morning"]):
            keywords.append("golden hour")
        elif any(word in combined for word in ["dusk", "sunset", "evening"]):
            keywords.append("warm lighting")
        elif any(word in combined for word in ["night", "midnight", "dark"]):
            keywords.append("night scene")

        # Mood-related
        if any(word in combined for word in ["happy", "joy", "celebrate", "laugh"]):
            keywords.append("bright colors")
        elif any(word in combined for word in ["sad", "cry", "mourn", "grief"]):
            keywords.append("muted colors")
        elif any(word in combined for word in ["tense", "danger", "threat", "fear"]):
            keywords.append("dramatic lighting")
        elif any(word in combined for word in ["peaceful", "calm", "serene"]):
            keywords.append("soft lighting")

        # Setting-related
        if any(word in combined for word in ["forest", "woods", "tree"]):
            keywords.append("natural lighting")
        elif any(word in combined for word in ["castle", "palace", "throne"]):
            keywords.append("grand architecture")
        elif any(word in combined for word in ["city", "street", "market"]):
            keywords.append("urban environment")

        return keywords

    def format_for_prompt(self, ctx: StyleContext) -> str:
        """Format style context for inclusion in LLM prompt.

        Args:
            ctx: The style context to format.

        Returns:
            Formatted string for prompt injection.
        """
        parts = []

        parts.append("=== Visual Continuity Context ===")

        # Characters
        if ctx.characters:
            parts.append("\nCharacters in this scene:")
            for name, desc in ctx.characters.items():
                parts.append(f"  - {name}: {desc}")

        # Setting
        if ctx.setting:
            parts.append(f"\nSetting: {ctx.setting}")
        if ctx.time_of_day:
            parts.append(f"Time: {ctx.time_of_day}")
        if ctx.mood:
            parts.append(f"Mood: {ctx.mood}")

        # Previous shot
        if ctx.previous_shot_summary:
            parts.append(f"\nPrevious shot: {ctx.previous_shot_summary}")
            parts.append("(Maintain visual continuity with the previous shot)")

        # Style hints
        if ctx.style_keywords:
            parts.append(f"\nStyle hints: {', '.join(ctx.style_keywords)}")

        parts.append("\n=================================")

        return "\n".join(parts)

    def get_video_generation_hints(self, ctx: StyleContext) -> dict:
        """Get hints for video generation from style context.

        Args:
            ctx: The style context.

        Returns:
            Dictionary of hints for video generator.
        """
        hints = {
            "seed": ctx.seed,
            "style_suffix": "",
        }

        # Build style suffix for video prompt
        suffixes = []
        
        if ctx.time_of_day:
            suffixes.append(ctx.time_of_day)
        if ctx.mood:
            suffixes.append(f"{ctx.mood} mood")
        if ctx.style_keywords:
            suffixes.extend(ctx.style_keywords[:3])  # Limit to top 3

        if suffixes:
            hints["style_suffix"] = ", ".join(suffixes)

        return hints


class MockStyleContextManager:
    """Mock style context manager for testing/debug mode."""

    def __init__(self, **kwargs):
        """Initialize mock manager."""
        pass

    def register_character_appearance(self, name: str, description: str, scene_id: int) -> None:
        """No-op."""
        pass

    def register_scene_setting(self, scene_id: int, **kwargs) -> None:
        """No-op."""
        pass

    def record_shot(self, scene_id: int, shot_index: int, summary: str, visual_style: str = "") -> None:
        """No-op."""
        pass

    def build_context_for_shot(self, scene, shot_index: int) -> StyleContext:
        """Return basic context."""
        return StyleContext(
            scene_id=scene.id,
            shot_index=shot_index,
        )

    def format_for_prompt(self, ctx: StyleContext) -> str:
        """Return empty string."""
        return ""

    def get_video_generation_hints(self, ctx: StyleContext) -> dict:
        """Return basic hints."""
        return {"seed": None, "style_suffix": ""}
