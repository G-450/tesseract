# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Welford's online algorithm for stable variance/std dev calculation.
- Exact serialized representation duplicate detection.
- PyMatGen compatibility gateway check (`verify_pymatgen.py`).
- Formal decision document regarding `material_id` limitation (`05_DECISIONS.md`).

### Fixed
- `pytest` path resolution (`pythonpath = ["."]`).
- `mypy` namespace collisions by adding `__init__.py` to `tests`.
- `flake8` line length and spacing violations.
- Decimal serialization issues in `ijson` by enabling `use_float=True`.

## [0.2.0] - 2026-08-01

### Added
- Added `ijson` to `requirements.txt` for streaming large JSON files.
- Added `configs/dataset.yaml` for configurable dataset paths.
- Added `src.data.discovery` for memory-safe dataset fingerprinting and discovery.
- Added `src.data.validator` for incremental dataset validation and online statistics.
- Added `src.data.cache` for atomic metadata caching.
- Added `src.cli.dataset_cli` providing `validate` and `info` commands.
- Added unit and integration tests in `tests/data` using synthetic JSON fixtures.

## [0.1.0] - 2026-08-01

### Added
- Initialized Git repository.
- Created project directory and documentation structure.
- Created Python project configuration (`pyproject.toml`, `requirements.txt`).
- Configured development tooling (black, isort, flake8, mypy, pytest).
- Created professional `README.md`, `LICENSE`, `PROJECT_STATUS.md`, `NEXT_TASK.md`, and `CHANGELOG.md`.
- Set up GitHub Actions CI workflow for linting, type-checking, formatting, and testing.
