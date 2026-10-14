# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-11

### Fixed
- **Checkpoint loading under PyTorch 2.6+:** `Evaluator.load_model()` (Phase 7) failed to load any real checkpoint on PyTorch 2.6 or newer, where `torch.load` defaults to `weights_only=True` and rejects the pickled `CGCNNConfig` stored in a checkpoint's hyperparameters. This broke `evaluate_cli.py` and `explain_cli.py`, which both load models through the `Evaluator`; the existing tests missed it because they mock the loader. The allowlist introduced for Phase 9 now lives in `src/utils/checkpoints.py` (PRD §10.10) and is shared by the evaluation and inference subsystems, so the permitted globals are defined in exactly one place. Covered by a regression test that reproduces the 2.6 default through Lightning's own loader.
- Removed a `bash.exe.stackdump` crash artifact that was committed by accident, and added `*.stackdump` to `.gitignore`.
- **Config validation (PRD §122):** `dataset.yaml` and `model.yaml` now reject unknown keys with `UnknownConfigKeyError` instead of a raw `TypeError`, matching `experiment.yaml`. Validation covers top-level section keys and nested section keys.
- **Split seed bug:** `CrystalDataModule` now accepts a `seed` parameter instead of hardcoding 42. `train_cli.py` and `report_cli.py` forward `experiment.seed` so the configured seed actually controls the train/val/test split. Default remains 42 for backward compatibility.

### Added
- **Phase 11: Final Hardening and v1.0 Release.**
  - **Unified CLI (`src/cli/main_cli.py`):** Single `project <command>` entry point (PRD §121) dispatching to `dataset`, `preprocess`, `train`, `evaluate`, `explain`, `predict`, `predict-batch`, `report`, `clean-cache` and `doctor`. Registered as `[project.scripts]` in `pyproject.toml`.
  - **`project doctor`:** Environment health check — Python version, core library imports with versions, CUDA availability, config files, dataset presence, graph cache status.
  - **`project clean-cache`:** Deletes cached `.pt` shards and `manifest*.json` files from the processed directory, with `--dry-run` support.
  - **End-to-end integration tests (`tests/test_integration_e2e.py`):** 12 tests covering the full pipeline on a synthetic fixture (train → evaluate → predict → report), config validation rejection, split seed determinism, and unified CLI smoke tests (PRD §111).
  - Version bumped to `1.0.0` in `pyproject.toml`.
  - 169 tests pass; `black`, `isort`, `flake8` clean.

- **Phase 10: Visualization and Reporting:** Fully implemented per PRD §17–§19, §94–§100.
  - `configs/experiment.yaml` and `src/config/experiment_config.py`: new `ExperimentConfig` defining run identity (`name`, `run_id`), the global `seed`, the artifact root (`experiments_dir`, `runs_dirname`), the published baseline MAE and the log level. Unknown keys raise `UnknownConfigKeyError` rather than being silently ignored (PRD §122), and a malformed file raises `ConfigFileError` rather than falling back to defaults.
  - **Run-ID concept:** a run is identified by `--run-id`, by `experiment.run_id`, or by a generated `"<name>_<UTC timestamp>"` such as `cgcnn_mp_e_form_20261010T153000Z`. IDs are sanitised to a single path segment, so a run can never write outside the runs root.
  - `src/utils/run_layout.py`: `RunLayout` defines and creates the one canonical layout under `experiments/runs/<run_id>/` — `config/`, `checkpoints/`, `logs/`, `plots/`, `explanations/`, `predictions/`, plus `metrics.json`, `predictions.csv`, `summary.md`, `metadata.json` and `report.md` (PRD §18, §19, §98). Creating a run refuses to clobber an existing directory unless `--overwrite` is passed, and `snapshot_configs()` copies the configurations the run actually used into `config/`.
  - `src/utils/provenance.py`: `collect_provenance()` records everything PRD §17 requires — git commit, branch and dirty flag, UTC timestamp, seed, configuration snapshot, library versions, platform and CPU, CUDA/cuDNN version and GPU names, dataset fingerprint and record counts read from the metadata cache and graph manifest, checkpoint SHA-256 and model version. Every source degrades to `None` rather than aborting the run, and `load_provenance()` reads a record back.
  - `src/evaluation/experiment_report.py`: `ExperimentReporter` renders the twelve-section Markdown report (metadata, dataset, configuration, training summary, evaluation metrics, benchmark comparison, visualizations, explainability, predictions, artifacts, conclusions, reproduction command). `summarize_training_log()`, `summarize_explanations()` and `summarize_predictions()` collect the inputs from disk. Rendering reads nothing from the clock or environment, so regenerating a report from the same run inputs is byte-identical — the Phase 10 reproducibility criterion. The reporter links figures it is given and never generates them, so evaluation keeps no dependency on visualization (PRD §11).
  - `src/visualization/plots.py`: added `plot_prediction_histogram()` and `plot_learning_rate()`, completing the figure set PRD §95 requires. Added an explicit `PUBLICATION_RC_PARAMS` style applied at import, so no figure relies on matplotlib defaults (PRD §97), and every series is now distinguished by both colour and dash pattern so figures stay readable in greyscale (PRD §94).
  - `src/cli/report_cli.py`: the `project report` entry point (PRD §121). Evaluates a checkpoint, exports metrics and per-sample predictions, generates every figure, collects provenance and writes `report.md`. `--report-only` re-renders the report from artifacts already on disk, reusing the recorded provenance so the output is unchanged; `--copy-logs` and `--copy-checkpoint` fold external artifacts into the run.
  - `src/cli/train_cli.py`: now writes into the run layout (checkpoints, CSV and TensorBoard logs, config snapshots) instead of shared top-level directories, seeds every RNG from `experiment.seed` via `seed_everything` (PRD §58), and adds `LearningRateMonitor` so the learning rate is logged and its required figure can be generated.
  - `src/evaluation/reporter.py`: `Reporter.export()` and `export_summary_md()` accept `artifact_links`, so the summary can point into the run's subdirectories instead of assuming a flat layout. The default links are unchanged.
  - `experiments/README.md`: documents the canonical layout, the run-ID rules, the command sequence for producing a run, and each legacy top-level directory (`cgcnn_logs/`, `cgcnn_csv_logs/`, `cgcnn_full_history/`, `checkpoints/`, `plots/`) with its origin and limitations.
  - `tests/test_reporting.py`: 45 unit and integration tests covering configuration validation and run-ID resolution, the run layout and its overwrite guard, provenance collection and round-tripping, the new figures, the artifact summaries, report completeness, byte-level reproducibility, graceful degradation when artifacts are absent, and `report_cli.py` including `--report-only`.
  - Verified end to end on the real 100-epoch checkpoint (`legacy_100_epoch`): 13,276 test graphs, MAE 0.0492 eV/atom, RMSE 0.0881 eV/atom, R² 0.9943, bias -0.0019 eV/atom, with five Integrated Gradients explanations and three batch predictions folded into the same run directory.

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
