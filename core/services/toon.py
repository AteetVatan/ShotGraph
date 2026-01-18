"""TOON (Token-Oriented Object Notation) codec for LLM communication.

TOON is a simpler, more token-efficient alternative to JSON for
structured LLM outputs. It uses indentation and minimal punctuation.

This module provides a thin wrapper around the PyPI `toons` package,
which is a fast Rust-based parser/serializer for TOON format.

Example TOON format:
```
scenes [3]:
1:
  summary: Hero meets mentor
  text: "The young hero walked..."
2:
  summary: The journey begins
  text: "They set off at dawn..."
```
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TOONCodec:
    """Encoder/decoder for TOON (Token-Oriented Object Notation) format.

    TOON is designed to be:
    - More token-efficient than JSON
    - Easier for LLMs to generate correctly
    - Simple to parse with fallback to JSON

    This implementation uses the fast Rust-based `toons` package from PyPI.
    """

    def decode(self, toon_text: str) -> dict[str, Any]:
        """Parse TOON format to dictionary.

        Args:
            toon_text: TOON-formatted string.

        Returns:
            Parsed dictionary.

        Raises:
            ValueError: If parsing fails or toons package is not installed.
        """
        try:
            import toons  # Lazy import for debug mode compatibility
        except ImportError as e:
            raise ValueError(
                "toons package not installed. Install with: pip install toons>=0.1.0"
            ) from e

        try:
            return toons.loads(toon_text)
        except Exception as e:
            # Wrap any exception from toons in ValueError for compatibility
            raise ValueError(f"Failed to parse TOON format: {e}") from e
