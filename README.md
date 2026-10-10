# Explainable Graph Neural Network Pipeline for Formation Energy Prediction

This project implements a complete, reproducible, modular, and explainable machine learning pipeline capable of predicting the **formation energy per atom** of crystalline materials directly from their crystal structures. It transforms crystal structures into graph representations and trains a **Crystal Graph Convolutional Neural Network (CGCNN)** to learn structure–property relationships from Density Functional Theory (DFT) derived datasets.

## Key Features

* **End-to-End Pipeline**: Automates the workflow from dataset acquisition to prediction and evaluation.
* **Graph Representation**: Converts crystal structures into graphs capturing atomic environments and connectivity without manual feature engineering.
* **Explainability**: Applies Explainable AI (XAI) techniques to provide qualitative interpretation of model predictions by identifying influential atoms and interactions.
* **Reproducibility**: Emphasizes configuration over hardcoding, strict random seed management, and deterministic execution.
* **Modular Architecture**: Built as a research framework where dataset processing, graph construction, training, evaluation, and explainability are decoupled and independently extensible.

## Quick Start

### Installation

It is recommended to use Python 3.12 and a virtual environment.

```bash
# Clone the repository
git clone <repository_url>
cd <repository_directory>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Alternatively, install as an editable package with dev dependencies:
pip install -e .[dev]
```

### Running an Experiment

Every run is identified by a **run ID** and writes all of its artifacts into
`experiments/runs/<run_id>/`, so runs never overwrite one another. The run ID
comes from `--run-id`, from `experiment.run_id` in `configs/experiment.yaml`,
or is generated as `<name>_<UTC timestamp>`.

```bash
# Prepare the dataset and build crystal graphs
python src/cli/dataset_cli.py validate
python src/cli/dataset_cli.py build-graphs

# Train. Creates experiments/runs/<run_id>/{checkpoints,logs,config}/
python src/cli/train_cli.py --run-id my_run

# Optional: explanations and predictions for external structures
python src/cli/explain_cli.py \
    --checkpoint experiments/runs/my_run/checkpoints/best.ckpt \
    --output-dir experiments/runs/my_run/explanations
python src/cli/predict_cli.py predict-batch \
    --checkpoint experiments/runs/my_run/checkpoints/best.ckpt \
    --input-dir path/to/cifs \
    --output-dir experiments/runs/my_run/predictions

# Evaluate, plot, record provenance and render the experiment report
python src/cli/report_cli.py \
    --run-id my_run \
    --checkpoint experiments/runs/my_run/checkpoints/best.ckpt

# Re-render report.md from artifacts already on disk
python src/cli/report_cli.py --run-id my_run --report-only
```

`experiments/runs/<run_id>/report.md` is the primary artifact for reviewing a
run: it collects provenance, dataset fingerprint, configuration, training
summary, test metrics, the published-baseline comparison, every figure and the
explainability and prediction artifacts. See
[`experiments/README.md`](experiments/README.md) for the full layout.

## Directory Structure

* `configs/` - YAML configuration files: `dataset.yaml`, `model.yaml` and
  `experiment.yaml` (run identity, seed and artifact layout)
* `datasets/` - Storage for raw datasets, processed data, cached graphs, and metadata
* `experiments/` - One directory per run under `experiments/runs/<run_id>/`
  (checkpoints, logs, plots, explanations, predictions, metrics, report);
  see [`experiments/README.md`](experiments/README.md)
* `notebooks/` - Jupyter notebooks for exploratory data analysis and demonstrations
* `outputs/` - General pipeline outputs
* `scripts/` - Standalone utility and execution scripts
* `src/` - The core application source code containing modules for config, data, graphs, models, training, evaluation, explainability, inference, visualization, utils, and cli.
* `tests/` - Automated test suite
* `docs/` - Project documentation including PRD, Architecture, and guides.

## Development

We use `black` for formatting, `isort` for import sorting, `flake8` for linting, and `mypy` for static type checking. Tests are written using `pytest`.

```bash
# Run tests
pytest

# Format code
black src tests
isort src tests

# Run linters and type checkers
flake8 src tests
mypy src
```
