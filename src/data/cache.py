import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.config.dataset_config import DatasetConfig


def _get_metadata_path(config: DatasetConfig) -> str:
    return os.path.join(config.metadata_dir, "dataset_metadata.json")


def save_metadata(
    config: DatasetConfig, metadata: Dict[str, Any], fingerprint: str
) -> None:
    """
    Saves dataset metadata to the cache directory atomically.

    Args:
        config: The dataset config.
        metadata: The metadata to save.
        fingerprint: The SHA-256 fingerprint of the dataset.
    """
    os.makedirs(config.metadata_dir, exist_ok=True)

    metadata_path = _get_metadata_path(config)
    temp_path = f"{metadata_path}.{uuid.uuid4().hex}.tmp"

    full_metadata = {
        "dataset_fingerprint": fingerprint,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": metadata,
    }

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2)

        # Atomic replace
        os.replace(temp_path, metadata_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_metadata(config: DatasetConfig, fingerprint: str) -> Optional[Dict[str, Any]]:
    """
    Loads dataset metadata if the fingerprint matches.

    Args:
        config: The dataset config.
        fingerprint: The current dataset fingerprint.

    Returns:
        The cached stats if valid, else None.
    """
    metadata_path = _get_metadata_path(config)

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            full_metadata = json.load(f)

        if full_metadata.get("dataset_fingerprint") == fingerprint:
            stats = full_metadata.get("stats")
            if isinstance(stats, dict):
                return stats
    except (json.JSONDecodeError, OSError):
        pass

    return None
