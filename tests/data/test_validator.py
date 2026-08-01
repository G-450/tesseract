from src.config.dataset_config import DatasetConfig
from src.data.validator import DatasetValidator
from tests.fixtures.data_fixtures import create_synthetic_dataset


def test_validator_valid_dataset(tmp_path):
    f = tmp_path / "valid.json"
    create_synthetic_dataset(str(f), valid=True)

    config = DatasetConfig(raw_data_path=str(f))
    validator = DatasetValidator(config)
    stats = validator.validate(str(f))

    assert stats["validation_status"] == "PASS"
    assert stats["total_records"] == 2
    assert stats["valid_records"] == 2
    assert stats["invalid_records"] == 0
    assert stats["target_mean"] == 2.0
    assert stats["target_min"] == 1.5
    assert stats["target_max"] == 2.5
    assert stats["target_std"] == 0.5


def test_validator_missing_fields(tmp_path):
    f = tmp_path / "missing.json"
    create_synthetic_dataset(str(f), valid=False, missing_field=True)

    config = DatasetConfig(raw_data_path=str(f))
    validator = DatasetValidator(config)
    stats = validator.validate(str(f))

    assert stats["validation_status"] == "FAIL"
    assert stats["invalid_records"] == 1
    assert stats["missing_fields"] == 1


def test_validator_invalid_target(tmp_path):
    f = tmp_path / "invalid_target.json"
    create_synthetic_dataset(str(f), valid=False, invalid_target=True)

    config = DatasetConfig(raw_data_path=str(f))
    validator = DatasetValidator(config)
    stats = validator.validate(str(f))

    assert stats["validation_status"] == "FAIL"
    assert stats["invalid_records"] == 1
    assert stats["invalid_targets"] == 1


def test_validator_nan_target(tmp_path):
    f = tmp_path / "nan_target.json"
    create_synthetic_dataset(str(f), valid=False, nan_target=True)

    config = DatasetConfig(raw_data_path=str(f))
    validator = DatasetValidator(config)
    stats = validator.validate(str(f))

    assert stats["validation_status"] == "FAIL"
    assert stats["invalid_records"] == 1
    assert stats["invalid_targets"] == 1


def test_validator_malformed_structure(tmp_path):
    f = tmp_path / "malformed.json"
    create_synthetic_dataset(
        str(f), valid=False, missing_field=False, invalid_target=False, nan_target=False
    )

    config = DatasetConfig(raw_data_path=str(f))
    validator = DatasetValidator(config)
    stats = validator.validate(str(f))

    assert stats["validation_status"] == "FAIL"
    assert stats["invalid_records"] == 1
    assert stats["malformed_structures"] == 1
