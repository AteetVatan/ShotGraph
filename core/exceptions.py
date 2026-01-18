"""Domain exceptions for ShotGraph."""


class ShotGraphError(Exception):
    """Base exception for all ShotGraph errors."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class LLMParseError(ShotGraphError):
    """Raised when LLM returns an unparseable response."""

    def __init__(self, message: str, *, raw_response: str | None = None):
        super().__init__(message, context={"raw_response": raw_response})
        self.raw_response = raw_response


class VideoGenerationError(ShotGraphError):
    """Raised when video generation fails."""

    def __init__(self, message: str, *, shot_id: int | None = None):
        super().__init__(message, context={"shot_id": shot_id})
        self.shot_id = shot_id


class TTSGenerationError(ShotGraphError):
    """Raised when TTS audio generation fails."""

    def __init__(self, message: str, *, text: str | None = None):
        super().__init__(message, context={"text": text})
        self.text = text


class MusicGenerationError(ShotGraphError):
    """Raised when music generation fails."""

    def __init__(self, message: str, *, scene_id: int | None = None):
        super().__init__(message, context={"scene_id": scene_id})
        self.scene_id = scene_id


class CompositionError(ShotGraphError):
    """Raised when video composition fails."""

    def __init__(self, message: str, *, stage: str | None = None):
        super().__init__(message, context={"stage": stage})
        self.stage = stage


class RetryExhaustedError(ShotGraphError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, *, attempts: int = 0):
        super().__init__(message, context={"attempts": attempts})
        self.attempts = attempts
