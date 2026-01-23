"""Device memory management utilities for ML model services."""

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "cuda"]

# Model file patterns for cache validation
MODEL_FILE_EXTENSIONS: frozenset[str] = frozenset({".safetensors", ".bin", ".json"})
MODEL_FILE_PATTERNS: tuple[str, ...] = ("model", "pytorch_model", "config")


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
        Path to HuggingFace hub cache directory (always absolute).
    """
    if settings and settings.hf_home and settings.hf_home.strip():
        # Resolve to absolute path to avoid path resolution issues in transformers
        return Path(settings.hf_home).resolve() / "hub"
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
    # Filter out .no_exist directories (HuggingFace cache artifacts from interrupted downloads)
    safetensors_files = [
        f for f in cache_path.rglob("*.safetensors")
        if ".no_exist" not in f.parts
    ]
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

    pytorch_files = [
        f for f in cache_path.rglob("pytorch_model*.bin")
        if ".no_exist" not in f.parts
    ]
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


def _has_model_files(cache_path: Path) -> bool:
    """Check if cache directory contains actual model files.

    Args:
        cache_path: Path to model cache directory.

    Returns:
        True if any model files exist (excluding .no_exist artifacts), False otherwise.
    """
    for item in cache_path.rglob("*"):
        if ".no_exist" in item.parts or not item.is_file():
            continue
        if item.suffix in MODEL_FILE_EXTENSIONS:
            if any(pattern in item.name.lower() for pattern in MODEL_FILE_PATTERNS):
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

    # Suppress warning if directory only contains artifacts (no actual model files)
    if not _has_model_files(cache_path):
        logger.debug(
            "Cache directory exists but contains only artifacts (no model files) for %s at %s",
            model_name,
            cache_path,
        )
    else:
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

    # Ensure cache_dir is a valid Path (fallback to default if somehow None)
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    # Convert to string and validate it's not empty
    # Only include cache_dir in kwargs if it's a valid non-empty string
    cache_dir_str = str(cache_dir).strip() if cache_dir else None
    if cache_dir_str == "":
        cache_dir_str = None

    # Clean up corrupted cache artifacts before validation
    cleanup_corrupted_cache(model_name, cache_dir, settings=settings)

    is_cached = is_model_cached(model_name, cache_dir, require_complete=True)

    if is_cached:
        logger.info(
            "Model %s found in cache, using local_files_only=True (faster, no download)",
            model_name,
        )
        try:
            # Only include cache_dir if it's valid
            load_kwargs_with_cache = {**load_kwargs}
            if cache_dir_str:
                load_kwargs_with_cache["cache_dir"] = cache_dir_str
            return load_func(
                model_name,
                local_files_only=True,
                **load_kwargs_with_cache,
            )
        except (OSError, AttributeError, ValueError) as e:
            # OSError: File not found / network issues
            # AttributeError: Cache incomplete (e.g., processor files missing)
            # ValueError: Invalid cache state or configuration
            logger.error(
                "Cache validation passed but model load failed for %s: %s. "
                "Cache may be corrupted or incomplete. Attempting download...",
                model_name,
                e,
            )

    logger.info(
        "Model %s not in cache or incomplete, downloading...",
        model_name,
    )
    # Only include cache_dir if it's valid
    load_kwargs_with_cache = {**load_kwargs}
    if cache_dir_str:
        load_kwargs_with_cache["cache_dir"] = cache_dir_str
    try:
        return load_func(
            model_name,
            local_files_only=False,
            **load_kwargs_with_cache,
        )
    except (OSError, AttributeError, ValueError, TypeError) as e:
        raise RuntimeError(
            f"Failed to load {model_name} from both cache and download. "
            f"Error: {e}. Try clearing cache at {cache_dir} or check network connection."
        ) from e


def cleanup_corrupted_cache(
    model_name: str,
    cache_dir: Path | None = None,
    *,
    settings: "Settings | None" = None,
) -> bool:
    """Remove corrupted cache entries for a model.

    This removes .no_exist directories and empty files that can cause
    validation warnings and download issues.

    Args:
        model_name: Full model name (e.g., "facebook/musicgen-medium").
        cache_dir: Optional cache directory (defaults to standard HF cache).
        settings: Optional settings for HF_HOME.

    Returns:
        True if cleanup was performed, False if no cleanup needed.
    """
    if cache_dir is None:
        cache_dir = get_hf_cache_dir(settings)

    cache_name = model_name.replace("/", "--")
    cache_path = cache_dir / f"models--{cache_name}"

    if not cache_path.exists():
        return False

    cleaned = False

    # Remove .no_exist directories (broken symlinks/references)
    no_exist_dirs = list(cache_path.rglob(".no_exist"))
    for no_exist_dir in no_exist_dirs:
        try:
            shutil.rmtree(no_exist_dir)
            logger.info("Removed corrupted .no_exist directory: %s", no_exist_dir)
            cleaned = True
        except OSError as e:
            logger.warning("Failed to remove .no_exist directory %s: %s", no_exist_dir, e)

    # Remove empty safetensors/pytorch files
    for pattern in ["*.safetensors", "pytorch_model*.bin"]:
        for file_path in cache_path.rglob(pattern):
            if ".no_exist" in file_path.parts:
                continue
            try:
                if file_path.stat().st_size == 0:
                    file_path.unlink()
                    logger.info("Removed empty cache file: %s", file_path)
                    cleaned = True
            except OSError as e:
                logger.warning("Failed to remove empty file %s: %s", file_path, e)

    return cleaned
