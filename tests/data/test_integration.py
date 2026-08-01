import subprocess
import sys

import yaml

from tests.fixtures.data_fixtures import create_synthetic_dataset


def test_cli_validate_and_info(tmp_path):
    # Setup dataset
    data_dir = tmp_path / "datasets" / "raw"
    data_dir.mkdir(parents=True)
    dataset_file = data_dir / "mp_e_form.json"
    create_synthetic_dataset(str(dataset_file), valid=True)

    # Setup config
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "dataset.yaml"

    config_content = {
        "dataset": {
            "raw_data_path": str(dataset_file),
            "metadata_dir": str(tmp_path / "datasets" / "metadata"),
            "processed_data_dir": str(tmp_path / "datasets" / "processed"),
            "require_structure": True,
            "require_target": True,
            "target_column": "e_form",
        }
    }

    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    # Run CLI validate
    result_val = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.dataset_cli",
            "--config",
            str(config_file),
            "validate",
        ],
        capture_output=True,
        text=True,
    )

    assert result_val.returncode == 0
    assert "Validation Report" in result_val.stdout
    assert "Status: PASS" in result_val.stdout

    # Run CLI info
    result_info = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.dataset_cli",
            "--config",
            str(config_file),
            "info",
        ],
        capture_output=True,
        text=True,
    )

    assert result_info.returncode == 0
    assert "Dataset Statistics" in result_info.stdout
    assert "Valid records: 2" in result_info.stdout
