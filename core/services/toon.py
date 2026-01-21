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

import re
import logging
from typing import Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TOONCodec:
    """Encoder/decoder for TOON (Token-Oriented Object Notation) format.

    TOON is designed to be:
    - More token-efficient than JSON
    - Easier for LLMs to generate correctly
    - Simple to parse with fallback to JSON

    This implementation uses the fast Rust-based `toons` package from PyPI.
    """

    def _validate_before_parse(self, toon_text: str) -> None:
        """Check for invalid standalone '-' lines before parsing.

        Args:
            toon_text: TOON-formatted string to validate.

        Raises:
            ValueError: If standalone '-' line is found with no following content (cannot be parsed even with strict=False).
        """
        lines = toon_text.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == '-':
                # Check if next non-empty line is properly indented (has content)
                has_content = False
                dash_indent = len(line) - len(line.lstrip())
                # Check next 10 lines after current line
                for j in range(i + 1, min(i + 11, len(lines))):
                    next_line = lines[j]
                    if not next_line.strip():
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    # If next line is more indented and has a colon (key-value), it's valid
                    if next_indent > dash_indent and ':' in next_line:
                        has_content = True
                        break
                    # If next line is at same or less indent, we've moved past this item
                    if next_indent <= dash_indent:
                        break
                
                if not has_content:
                    raise ValueError(
                        f"Invalid TOON: standalone '-' at line {i + 1} with no following content. "
                        "This cannot be parsed even with strict=False."
                    )

    def _repair_standalone_dashes(self, toon_text: str) -> str:
        """Repair standalone '-' items by converting to '-:' for object items.

        The toons library requires '-:' (dash-colon) for array items that contain
        objects/structs with multiple key-value pairs. A standalone '-' is only
        valid for primitive values.

        Also handles cases where '- key: value' format exists (from previous merges),
        converting them to proper '-:' format with key-value on next line.

        Args:
            toon_text: Normalized TOON text that may contain standalone dashes or merged dashes.

        Returns:
            Repaired TOON text with '-:' for object items.
        """
        lines = toon_text.split('\n')
        repaired_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            line_indent = len(line) - len(line.lstrip())
            
            # Check if this is a merged dash pattern like "- id: 1" or "- key: value"
            merged_dash_match = re.match(r'^(\s*)-\s+(\w+)\s*:\s*(.+)$', line)
            if merged_dash_match:
                # This is a merged dash that needs to be split into '-:' format
                indent = merged_dash_match.group(1)
                key = merged_dash_match.group(2)
                value = merged_dash_match.group(3)
                # Convert to '-:' format with key-value on next line
                repaired_lines.append(f"{indent}-:")
                # Add the key-value pair on the next line with proper indentation
                repaired_lines.append(f"{indent}  {key}: {value}")
                i += 1
                continue
            
            # Check if this is a standalone dash
            if line_stripped == '-':
                # Look ahead to see if this dash is followed by indented key-value pairs
                has_indented_content = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j]
                    if not next_line.strip():
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    # If next line is more indented and has a colon (key-value), it's an object
                    if next_indent > line_indent and ':' in next_line:
                        has_indented_content = True
                        break
                    # If we hit a line at same or less indent, stop looking
                    if next_indent <= line_indent:
                        break
                
                if has_indented_content:
                    # Convert '-' to '-:' for object items (toons library requirement)
                    repaired_lines.append(f"{' ' * line_indent}-:")
                else:
                    # No indented content, keep as-is (primitive value or will fail validation)
                    repaired_lines.append(line)
                i += 1
            else:
                repaired_lines.append(line)
                i += 1
        
        return '\n'.join(repaired_lines)

    def _normalize_toon_format(self, toon_text: str) -> str:
        """Normalize TOON format from LLM output to parser-compatible format.

        Converts numbered array items (1:, 2:, etc.) to dash-prefixed items (-)
        that the toons library expects. Also removes explicit array length declarations.
        Preserves relative indentation of content under each item.

        Args:
            toon_text: Raw TOON-formatted string from LLM.

        Returns:
            Normalized TOON string compatible with toons parser.
        """
        lines = toon_text.split('\n')
        normalized_lines = []
        array_indent_level = -1
        last_numbered_item_indent = -1
        last_dash_indent = -1

        for line in lines:
            line_stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())

            # Check if this line declares an array with length (e.g., "scenes [4]:")
            array_match = re.match(r'^(\s*)(\w+)\s*\[\d+\]\s*:\s*$', line)
            if array_match:
                indent = array_match.group(1)
                array_name = array_match.group(2)
                array_indent_level = len(indent)
                last_numbered_item_indent = -1
                last_dash_indent = -1
                normalized_lines.append(f"{indent}{array_name}:")
                continue

            # Check if line is a numbered array item (e.g., "1:", "2:", etc.)
            numbered_item_match = re.match(r'^(\s*)(\d+)\s*:\s*$', line)
            if numbered_item_match:
                item_indent = numbered_item_match.group(1)
                item_indent_len = len(item_indent)
                last_numbered_item_indent = item_indent_len

                # If item is at root or same level as array declaration, indent it under array
                if array_indent_level >= 0 and item_indent_len <= array_indent_level:
                    # Indent the dash item under the array (2 spaces standard)
                    last_dash_indent = array_indent_level + 2
                    normalized_lines.append(f"{' ' * last_dash_indent}-")
                else:
                    # Keep relative indentation
                    last_dash_indent = item_indent_len
                    normalized_lines.append(f"{item_indent}-")
                continue

            # Adjust indentation of content lines that follow numbered items
            if last_dash_indent >= 0 and last_numbered_item_indent >= 0 and line_stripped:
                # Check if this line is indented relative to the numbered item
                if current_indent > last_numbered_item_indent:
                    # Content was indented under numbered item - adjust to be relative to dash
                    relative_indent = current_indent - last_numbered_item_indent
                    new_indent = last_dash_indent + relative_indent
                    adjusted_line = ' ' * new_indent + line_stripped
                    normalized_lines.append(adjusted_line)
                    continue
                elif current_indent <= last_numbered_item_indent:
                    # Content is at same or less indent - new item or end of previous
                    # Reset context so this line is processed normally
                    last_dash_indent = -1
                    last_numbered_item_indent = -1

            # Reset array context if we hit a line with less indent than array
            if array_indent_level >= 0 and line_stripped:
                if current_indent <= array_indent_level and not re.match(r'^\s*\d+\s*:\s*$', line):
                    array_indent_level = -1
                    last_numbered_item_indent = -1
                    last_dash_indent = -1

            # Regular line - keep as is (preserves relative indentation)
            # Empty lines pass through unchanged and don't reset context
            normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def _fix_array_parsing(self, parsed: dict[str, Any], normalized_text: str) -> dict[str, Any]:
        """Fix array parsing where multiple '-:' items are collapsed into a single dict key.
        
        When the toons library parses multiple '-:' items at the same level, it treats them
        as duplicate dictionary keys, keeping only the last one. This method detects such
        cases and reconstructs the array by extracting items from the normalized text.
        
        Args:
            parsed: The result from toons.loads() which may have collapsed arrays.
            normalized_text: The normalized TOON text used for parsing.
            
        Returns:
            Dictionary with arrays properly reconstructed.
        """
        try:
            import toons
        except ImportError:
            # If toons is not available, return as-is
            return parsed
        
        result = parsed.copy()
        lines = normalized_text.split('\n')
        
        # Check each top-level key that might be an array
        for key, value in parsed.items():
            # If value is a dict with only a '-' or '-:' key, it might be a collapsed array
            if isinstance(value, dict) and len(value) == 1 and ('-' in value or '-:' in value):
                # Extract all array items from the normalized text
                array_items = self._extract_array_items(lines, key)
                
                # If we found multiple items, replace the dict with an array
                if len(array_items) > 1:
                    result[key] = array_items
                    logger.debug(
                        "Fixed collapsed array for key '%s': found %d items (was dict with 1 key)",
                        key, len(array_items)
                    )
        
        return result
    
    def _extract_array_items(self, lines: list[str], array_key: str) -> list[dict[str, Any]]:
        """Extract all array items for a given array key from normalized TOON text.
        
        Args:
            lines: Lines of the normalized TOON text.
            array_key: The key name of the array (e.g., 'scenes').
            
        Returns:
            List of dictionaries, one per array item.
        """
        try:
            import toons
        except ImportError:
            return []
        
        array_items = []
        array_key_indent = None
        current_item_lines = []
        in_array_context = False
        item_base_indent = None
        
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Detect array declaration (e.g., "scenes:")
            if stripped == f"{array_key}:":
                in_array_context = True
                array_key_indent = indent
                continue
            
            if not in_array_context or array_key_indent is None:
                continue
            
            # If we're past the array context (less or equal indent, not a dash), stop
            if (indent <= array_key_indent and 
                stripped and not (stripped.startswith('-') or stripped.startswith('-:'))):
                break
            
            # Detect array items ('-:' or '-' at the expected indent level)
            expected_item_indent = array_key_indent + 2
            is_item_marker = (indent == expected_item_indent and 
                            (stripped == '-:' or stripped == '-'))
            
            if is_item_marker:
                # Parse previous item if exists
                if current_item_lines:
                    item_dict = self._parse_item_from_lines(current_item_lines, toons)
                    if item_dict:
                        array_items.append(item_dict)
                
                # Start new item
                current_item_lines = []
                item_base_indent = expected_item_indent
            elif item_base_indent is not None:
                # Collect lines for current item (must be more indented than item marker)
                if indent > item_base_indent:
                    current_item_lines.append(line)
                elif indent == item_base_indent and stripped:
                    # New item at same level (shouldn't happen, but handle gracefully)
                    if current_item_lines:
                        item_dict = self._parse_item_from_lines(current_item_lines, toons)
                        if item_dict:
                            array_items.append(item_dict)
                    current_item_lines = []
                    item_base_indent = None
        
        # Parse last item
        if current_item_lines:
            item_dict = self._parse_item_from_lines(current_item_lines, toons)
            if item_dict:
                array_items.append(item_dict)
        
        return array_items
    
    def _parse_item_from_lines(self, lines: list[str], toons_module: Any) -> dict[str, Any] | None:
        """Parse a TOON item from lines using the toons library.
        
        Args:
            lines: Lines representing a single TOON item (with original indentation).
            toons_module: The imported toons module.
            
        Returns:
            Parsed dictionary or None if parsing fails.
        """
        if not lines:
            return None
        
        # Remove base indentation from all lines to make them parseable as a standalone TOON object
        # Find minimum indentation (excluding empty lines)
        min_indent = None
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if min_indent is None or indent < min_indent:
                    min_indent = indent
        
        if min_indent is None:
            return None
        
        # Remove base indentation from all lines
        normalized_lines = []
        for line in lines:
            if line.strip():
                normalized_lines.append(line[min_indent:])
            else:
                normalized_lines.append('')
        
        # Build a standalone TOON document for this item
        item_text = '\n'.join(normalized_lines)
        
        try:
            # Try parsing as a standalone TOON object
            item_dict = toons_module.loads(item_text, strict=False)
            if isinstance(item_dict, dict):
                # Extract the actual item (might be under '-' or '-:' key, or at root)
                if '-' in item_dict:
                    return item_dict['-']
                if '-:' in item_dict:
                    return item_dict['-:']
                # If no dash key, return the dict itself (might be flat structure)
                return item_dict
            return None
        except Exception:
            # Fallback to manual parsing
            return self._parse_item_manually(lines)
    
    def _parse_item_manually(self, lines: list[str]) -> dict[str, Any] | None:
        """Manually parse a TOON item from lines as fallback.
        
        Args:
            lines: Lines representing a single TOON item.
            
        Returns:
            Parsed dictionary or None if parsing fails.
        """
        item = {}
        base_indent = None
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('-'):
                continue
            
            indent = len(line) - len(line.lstrip())
            if base_indent is None:
                base_indent = indent
            
            # Only process lines at the base indent level (direct children of item)
            if indent == base_indent and ':' in stripped:
                parts = stripped.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"\'')
                    item[key] = value
        
        return item if item else None

    def decode(self, toon_text: str, *, strict: bool = False) -> dict[str, Any]:
        """Parse TOON format to dictionary.

        Args:
            toon_text: TOON-formatted string.
            strict: If False, allows some slightly malformed cases (e.g., blank lines in arrays).
                    Note: This will NOT fix a truly invalid standalone '-' item.

        Returns:
            Parsed dictionary.

        Raises:
            ValueError: If parsing fails, toons package is not installed, or invalid standalone '-' found.
        """
        try:
            import toons  # Lazy import for debug mode compatibility
        except ImportError as e:
            raise ValueError(
                "toons package not installed. Install with: pip install toons>=0.1.0"
            ) from e

        # Normalize the format before parsing
        normalized_text = self._normalize_toon_format(toon_text)
        logger.debug("Normalized TOON format (original length: %d, normalized: %d)",
                     len(toon_text), len(normalized_text))

        # Pre-parse guard: check normalized text for standalone '-' lines (cannot be fixed even with strict=False)
        # Check normalized text because normalization may create standalone dashes
        self._validate_before_parse(normalized_text)

        # Repair standalone '-' items by merging with next line if it's a key-value pair
        normalized_text = self._repair_standalone_dashes(normalized_text)

        try:
            res = toons.loads(normalized_text, strict=strict)
        except Exception as e:
            # Wrap any exception from toons in ValueError for compatibility
            raise ValueError(f"Failed to parse TOON format: {e}") from e

        # Fix array parsing issues where multiple '-:' items collapse into a single dict key
        res = self._fix_array_parsing(res, normalized_text)

        return res

    def encode(self, data: Union[dict[str, Any], "BaseModel"]) -> str:
        """Encode dict or Pydantic model to canonical TOON format.

        This guarantees valid TOON output from validated data. Use this to emit
        canonical TOON after validating through Pydantic models.

        Args:
            data: Dictionary or Pydantic model to encode.

        Returns:
            Canonical TOON string (guaranteed valid).

        Raises:
            ValueError: If toons package is not installed or encoding fails.
        """
        try:
            import toons  # Lazy import for debug mode compatibility
        except ImportError as e:
            raise ValueError(
                "toons package not installed. Install with: pip install toons>=0.1.0"
            ) from e

        # Convert Pydantic model to dict if needed
        if hasattr(data, "model_dump"):
            # Pydantic v2
            data = data.model_dump(mode="json")
        elif hasattr(data, "dict"):
            # Pydantic v1 fallback
            data = data.dict()

        try:
            return toons.dumps(data)
        except Exception as e:
            raise ValueError(f"Failed to encode TOON format: {e}") from e
