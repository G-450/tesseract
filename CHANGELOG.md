# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **Phase 9: Inference and Prediction:** Fully implemented per PRD §89–§92.
  - `src/inference/predictor.py`: `Predictor` loading a checkpoint in frozen `eval` mode and predicting formation energy (eV/atom) for structures outside the dataset. `PredictionResult` and `BatchPredictionResult` dataclasses, the latter carrying per-structure failures and an aggregate `summary()` (count, mean, min, max, std).
  - Preprocessing parity (PRD §90): inference builds graphs through the training `CrystalGraphBuilder` with the `GraphConfig` from `configs/dataset.yaml`. Verified bit-identical — `x`, `edge_index`, `edge_attr` and `edge_image` all match the processed shards training consumed for the same records.
  - Single prediction from a CIF file or an in-memory `pymatgen.Structure`; batch prediction over a directory or an explicit path list. Structures that fail to parse or predict are logged and skipped, never aborting the batch (PRD §91).
  - Batched forward passes with automatic per-structure fallback, so one bad graph cannot discard an entire batch. Batched and single-structure scoring agree to within 1e-5 eV/atom.
  - `src/inference/exporter.py`: `PredictionExporter` writing `predictions.csv` and `predictions.json` (predictions, run summary, skipped structures), plus `export_single()` for one-off CSV/JSON output.
  - `src/inference/exceptions.py`: `InferenceError` hierarchy — `ModelLoadError`, `StructureLoadError`, `PredictionError`, `PredictionExportError` — integrated with the project exception system (PRD §123).
  - `src/inference/__init__.py`: full public API with `__all__`.
  - `src/cli/predict_cli.py`: `predict` and `predict-batch` subcommands with `--checkpoint`, `--dataset-config`, `--model-config`, `--input` / `--input-dir`, `--output` / `--output-dir`, `--batch-size` and `--device` (PRD §121).
  - Inference depends only on `src.config`, `src.graphs` and `src.models` — never on the trainer, the datamodule or the processed dataset (PRD §89).
  - Checkpoint loading works under PyTorch 2.6+, where `torch.load` defaults to `weights_only=True`: the project's own config dataclasses (`CGCNNConfig`, `ModelConfig`, `OptimizerConfig`) are allowlisted for the duration of the load via `torch.serialization.safe_globals`, so pickled hyperparameters deserialize while every other global stays blocked. Falls back to a no-op on PyTorch versions without `safe_globals`.
  - `tests/test_inference.py`: 44 unit and integration tests covering the exception hierarchy, result dataclasses, training/inference preprocessing parity, checkpoint loading and freezing, structure discovery, failure isolation, batching equivalence, CSV/JSON export and both CLI subcommands.

- **Phase 8: Explainability Subsystem:** Fully implemented per PRD §82–§88.
  - `src/explainability/base.py`: `BaseExplainer` ABC with `explain()`, `explain_batch()`, `visualize()`, `export()` abstract methods; `ExplanationResult` dataclass with auto-computed `summary_statistics` (top node/edge idx, scores, mean, std, prediction error).
  - `src/explainability/exceptions.py`: `ExplainabilityError` hierarchy — `ExplainerInitializationError`, `SampleExplanationError`, `ExportError` — integrated with project exception system (PRD §123).
  - `src/explainability/integrated_gradients.py`: `IntegratedGradientsExplainer` using Captum. Hooks after `nn.Embedding` to operate on continuous space. Attributes both node embeddings and edge features simultaneously.
  - `src/explainability/gnn_explainer.py`: `GNNExplainerWrapper` with graceful fallback when `torch_geometric.explain` is unavailable (PRD §85, §88). Skips single-atom structures due to model input dimensions.
  - `src/explainability/exporter.py`: `Exporter` producing per-material `explanation.json` + `heatmap.png` under `<output_dir>/<sample_id>/` plus aggregated `summary.csv` (PRD §87).
  - `src/explainability/__init__.py`: Full public API with `__all__`.
  - `src/cli/explain_cli.py`: `explain_cli.py` entry point with `--checkpoint`, `--method`, `--num-samples`, `--output-dir`, `--device`, `--num-workers`, `--seed` flags (PRD §121).
  - GNNExplainer outputs are reproducible: the `--seed` (default 42) is re-applied per sample inside a forked RNG and recorded in each result's metadata. Bit-identical on CPU; float-level (~1e-7) noise only on CUDA.
  - `tests/test_explainability.py`: 20 unit and integration tests covering ABC enforcement, summary statistics, IG shapes, batch skip logic, visualize/export artifacts, exception attributes, Exporter directory structure.
  - `captum>=0.7.0` added to runtime dependencies.

- **Phase 3: Crystal Graph Construction and Serialization:** Fully audited, verified, and complete. 
  - Integrated `ijson`-backed pipeline for bounded graph streaming logic.
  - Implemented analytical `max_neighbors` cutoff with `pymatgen` bounded distances.
  - Fixed mypy annotations and explicitly deferred scalar distance recording (F-001).
  - Explicitly documented Graph Asymmetry via `max_neighbors` in `05_DECISIONS.md` (F-002).
  - Audited mathematical periodic boundary condition mapping via analytical tests.
  - Test suite running at 100% capacity via pytest/flake8/black/isort/mypy checks.
- Welford's online algorithm for stable variance/std dev calculation.
- Exact serialized representation duplicate detection.
- PyMatGen compatibility gateway check (`verify_pymatgen.py`).
- Formal decision document regarding `material_id` limitation (`05_DECISIONS.md`).

### Fixed
- `pytest` path resolution (`pythonpath = ["."]`).
- `mypy` namespace collisions by adding `__init__.py` to `tests`.
- `flake8` line length and spacing violations.
- Decimal serialization issues in `ijson` by enabling `use_float=True`.

## [0.3.0] - Phase 3 Completion

### Added
- `dataset_config.py:GraphConfig` configuration class for graph cutoff, max neighbors, and RBF settings.
- `src/graphs/features.py:GaussianDistance` module for PyTorch Geometric edge feature expansion.
- `src/graphs/builder.py:CrystalGraphBuilder` module mapping structure distances into deterministic graphs.
- `src/data/pipeline.py:GraphPipeline` orchestrator using `ijson` and sharded saving for constant-memory scaling.
- `dataset_cli.py build-graphs` command, providing an easy entry point with `--smoke-test` capabilities.
- Unit and integration tests for features, builder, and pipeline ensuring processing reliability.

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
