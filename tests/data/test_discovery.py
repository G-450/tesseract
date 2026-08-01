from typing import Any

import pytest

from src.config.dataset_config import DatasetConfig
from src.data.discovery import compute_fingerprint, discover_dataset


def test_discover_dataset_missing() -> None:  # type: ignore
    config = DatasetConfig(raw_data_path="nonexistent.json")
    with pytest.raises(FileNotFoundError):
        discover_dataset(config)


def test_discover_dataset_success(tmp_path: Any) -> None:  # type: ignore
    f = tmp_path / "dummy.json"
    f.write_text("{}")

    config = DatasetConfig(raw_data_path=str(f))
    result = discover_dataset(config)

    assert result["exists"] is True
    assert result["size_bytes"] == 2


def test_compute_fingerprint(tmp_path: Any) -> None:  # type: ignore
    f = tmp_path / "dummy.json"
    f.write_text("hello world")

    # sha256 of "hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert compute_fingerprint(str(f)) == expected
