import hashlib
import logging
import os
from typing import Any, Dict

from src.config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


def discover_dataset(config: DatasetConfig) -> Dict[str, Any]:
    """
    Checks if the dataset exists, is a file, is readable, and returns its size.

    Args:
        config: The dataset configuration.

    Returns:
        Dict containing path, size_bytes, and exists flag.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        PermissionError: If the dataset is not readable.
    """
    path = config.raw_data_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Path {path} exists but is not a regular file.")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"Dataset at {path} is not readable.")

    size_bytes = os.path.getsize(path)

    return {"path": path, "exists": True, "size_bytes": size_bytes, "is_readable": True}


def compute_fingerprint(filepath: str, chunk_size: int = 8192) -> str:
    """
    Computes a SHA-256 fingerprint for a file using chunked reading.
    This avoids loading the entire file into memory.

    Args:
        filepath: Path to the file.
        chunk_size: Bytes to read per chunk.

    Returns:
        The SHA-256 hex digest of the file.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()
