"""Protocol interfaces for ShotGraph services."""

from core.protocols.llm_client import ILLMClient
from core.protocols.music_generator import IMusicGenerator
from core.protocols.tts_generator import ITTSGenerator
from core.protocols.video_generator import IVideoGenerator

__all__ = [
    "ILLMClient",
    "IVideoGenerator",
    "ITTSGenerator",
    "IMusicGenerator",
]
