"""Device memory management utilities for ML model services."""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "cuda"]


def detect_device() -> DeviceType:
    """Detect the available compute device.

    Returns:
        "cuda" if GPU is available, "cpu" otherwise.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def cleanup_memory(device: DeviceType) -> None:
    """Clean up memory before generation (CPU-specific optimizations).

    Args:
        device: The compute device ("cpu" or "cuda").
    """
    try:
        import torch
        import gc

        if device == "cpu":
            gc.collect()
            # Limit CPU threads to prevent memory fragmentation
            torch.set_num_threads(min(4, torch.get_num_threads()))
        elif device == "cuda":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception:
        pass  # Non-critical optimization
