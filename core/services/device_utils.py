"""Device memory management utilities for ML model services."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from config.settings import Settings

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


def get_hf_cache_dir(settings: "Settings | None" = None) -> Path:
    """Get HuggingFace cache directory path.

    Args:
        settings: Optional settings object (for custom HF_HOME).

    Returns:
        Path to HuggingFace hub cache directory.
    """
    if settings and settings.hf_home:
        return Path(settings.hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _validate_sharded_cache(cache_path: Path, model_name: str) -> bool:
    """Validate sharded model cache completeness.

    Args:
        cache_path: Path to model cache directory.
        model_name: Model name for logging.

    Returns:
        True if all shards exist, False otherwise.
    """
    index_file = cache_path / "refs" / "main" / "model.safetensors.index.json"
    if not index_file.exists():
        return False

    try:
        with open(index_file, "r") as f:
            index_data = json.load(f)

        if "weight_map" not in index_data:
            return False

        blobs_dir = cache_path / "blobs"
        for shard_path in index_data["weight_map"].values():
            blob_file = blobs_dir / shard_path.split("/")[-1]
            if not blob_file.exists() or blob_file.stat().st_size == 0:
                logger.warning(
                    "Cache incomplete: missing or empty shard %s for %s",
                    blob_file.name,
                    model_name,
                )
                return False

        return True
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning(
            "Cache index corrupted for %s: %s",
            model_name,
            e,
        )
        return False


def _validate_single_file_cache(cache_path: Path, model_name: str) -> bool:
    """Validate single-file model cache completeness.

    Args:
        cache_path: Path to model cache directory.
        model_name: Model name for logging.

    Returns:
        True if valid model files exist, False otherwise.
    """
    safetensors_files = list(cache_path.rglob("*.safetensors"))
    if safetensors_files:
        for sf_file in safetensors_files:
            if sf_file.stat().st_size == 0:
                logger.warning(
                    "Cache corrupted: empty file %s for %s",
                    sf_file,
                    model_name,
                )
                return False
        return True

    pytorch_files = list(cache_path.rglob("pytorch_model*.bin"))
    if pytorch_files:
        for pt_file in pytorch_files:
            if pt_file.stat().st_size == 0:
                logger.warning(
                    "Cache corrupted: empty file %s for %s",
                    pt_file,
                    model_name,
                )
                return False
        return True

    return False


def is_model_cached(
    model_name: str,
    cache_dir: Path | None = None,
    *,
    require_complete: bool = True,
) -> bool:
    """Check if a HuggingFace model is completely cached locally.

    This validates that the cache is not just present, but COMPLETE.
    Incomplete caches cause re-downloads and disk quota issues.

    Args:
        model_name: Full model name (e.g., "stabilityai/stable-video-diffusion-img2vid").
        cache_dir: Optional cache directory (defaults to standard HF cache).
        require_complete: If True, validate cache completeness (recommended).

    Returns:
        True if model is cached and (if require_complete) complete, False otherwise.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    cache_name = model_name.replace("/", "--")
    cache_path = cache_dir / f"models--{cache_name}"

    if not cache_path.exists():
        return False

    if not require_complete:
        return True

    if _validate_sharded_cache(cache_path, model_name):
        return True

    if _validate_single_file_cache(cache_path, model_name):
        return True

    logger.warning(
        "Cache directory exists but incomplete for %s at %s",
        model_name,
        cache_path,
    )
    return False


def load_model_with_cache_check(
    model_name: str,
    load_func: Callable[..., Any],
    *,
    cache_dir: Path | None = None,
    settings: "Settings | None" = None,
    **load_kwargs: Any,
) -> Any:
    """Load HuggingFace model with intelligent cache checking.

    This function:
    1. Checks if model is cached and complete
    2. Uses local_files_only=True if cache is valid (prevents re-download)
    3. Falls back to download if cache is missing/corrupted
    4. Handles errors gracefully

    Args:
        model_name: Full model name.
        load_func: Function to call (e.g., StableVideoDiffusionPipeline.from_pretrained).
        cache_dir: Optional cache directory.
        settings: Optional settings for HF_HOME.
        **load_kwargs: Additional kwargs to pass to load_func.

    Returns:
        Loaded model/pipeline.

    Raises:
        OSError: If cache is missing and local_files_only=True was used.
        Exception: Other model loading errors.
    """
    if cache_dir is None:
        cache_dir = get_hf_cache_dir(settings)

    is_cached = is_model_cached(model_name, cache_dir, require_complete=True)

    if is_cached:
        logger.info(
            "Model %s found in cache, using local_files_only=True (faster, no download)",
            model_name,
        )
        try:
            return load_func(
                model_name,
                local_files_only=True,
                cache_dir=str(cache_dir) if cache_dir else None,
                **load_kwargs,
            )
        except OSError as e:
            logger.error(
                "Cache validation passed but model load failed for %s: %s. "
                "Cache may be corrupted. Attempting download...",
                model_name,
                e,
            )

    logger.info(
        "Model %s not in cache or incomplete, downloading...",
        model_name,
    )
    return load_func(
        model_name,
        local_files_only=False,
        cache_dir=str(cache_dir) if cache_dir else None,
        **load_kwargs,
    )
