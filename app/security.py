"""Security middleware and dependencies for ShotGraph API."""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)

# API Key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache
def _get_settings() -> "Settings":
    """Get cached settings (avoid circular import)."""
    from config.settings import Settings

    return Settings()


async def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str:
    """Verify the API key from request header.

    If API key authentication is disabled, returns "anonymous".
    Otherwise, validates the provided key against the configured key.

    Args:
        api_key: The API key from the X-API-Key header.

    Returns:
        The validated API key or "anonymous" if auth is disabled.

    Raises:
        HTTPException: If API key is missing or invalid.
    """
    settings = _get_settings()

    # If API key auth is disabled, allow all requests
    if not settings.api_key_enabled:
        return "anonymous"

    # API key auth is enabled - validate
    if not api_key:
        logger.warning("API key missing in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    logger.debug("API key validated successfully")
    return api_key


def get_rate_limiter():
    """Get the rate limiter instance.

    Returns:
        Configured Limiter instance, or None if slowapi is not available.
    """
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address

        settings = _get_settings()
        return Limiter(
            key_func=get_remote_address,
            default_limits=[f"{settings.rate_limit_per_minute}/minute"],
        )
    except ImportError:
        logger.warning("slowapi not installed, rate limiting disabled")
        return None


def setup_rate_limiter(app):
    """Configure rate limiter on the FastAPI app.

    Args:
        app: The FastAPI application instance.
    """
    limiter = get_rate_limiter()
    if limiter is None:
        return

    try:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded

        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("Rate limiting enabled")
    except ImportError:
        logger.warning("Could not configure rate limiter")


def rate_limit(limit: str | None = None):
    """Decorator factory for rate limiting endpoints.

    Args:
        limit: Rate limit string (e.g., "10/minute"). If None, uses default.

    Returns:
        Rate limit decorator if slowapi is available, otherwise passthrough.
    """
    limiter = get_rate_limiter()
    if limiter is None:
        # Return a no-op decorator
        def passthrough(func):
            return func

        return passthrough

    if limit:
        return limiter.limit(limit)
    return limiter.limit(_get_settings().rate_limit_per_minute.__str__() + "/minute")


class SecurityDependency:
    """Combined security dependency for protected endpoints.

    Validates API key and tracks the authenticated client.
    """

    def __init__(self, require_auth: bool = True):
        """Initialize the security dependency.

        Args:
            require_auth: Whether to require authentication.
        """
        self.require_auth = require_auth

    async def __call__(
        self,
        request: Request,
        api_key: str = Depends(verify_api_key),
    ) -> str:
        """Validate security for the request.

        Args:
            request: The incoming request.
            api_key: The validated API key.

        Returns:
            The client identifier (API key or "anonymous").
        """
        # Store client info for logging/tracking
        request.state.client_id = api_key
        return api_key


# Convenience dependency instances
require_auth = SecurityDependency(require_auth=True)
optional_auth = SecurityDependency(require_auth=False)
