# `experiments/`

Every artifact produced by a training, evaluation, explainability or
inference run lives here. The contents are generated, not source, and are
excluded from version control (`.gitignore`) except for this file.

> **Retention policy: nothing in `experiments/` is ever deleted.**
>
> The directory is append-only. Runs are added; they are never removed,
> overwritten or pruned, and neither are the legacy directories described at
> the end of this file. Experiment artifacts are the record of what was
> actually run — `cgcnn_full_history/metrics.csv` and
> `checkpoints/cgcnn-epoch=99-val_loss=0.0070.ckpt` are the only surviving
> trace of the one real 100-epoch training run, and the dataset and compute
> needed to reproduce them are not cheap.
>
> The single exception is the explicit `report_cli.py --overwrite` flag,
> which replaces one named run directory on request. It is never implied by
> any other command: creating a run that already exists fails instead.

## Canonical layout (Phase 10)

One run = one directory, named after its **run ID**. Runs never overwrite one
another (PRD §18).

```text
experiments/
└── runs/
    └── <run_id>/
        ├── config/              # snapshots of the configs the run used
        │   ├── dataset.yaml
        │   ├── model.yaml
        │   └── experiment.yaml
        ├── checkpoints/         # best.ckpt, last.ckpt, cgcnn-epoch=NN-*.ckpt
        ├── logs/                # csv/ and tensorboard/ logger output
        ├── plots/               # publication-quality figures (PNG, 300 DPI)
        │   ├── parity_plot.png
        │   ├── residual_plot.png
        │   ├── prediction_histogram.png
        │   ├── loss_curve.png
        │   └── learning_rate.png
        ├── explanations/        # summary.csv + <sample_id>/{explanation.json,heatmap.png}
        ├── predictions/         # predictions.csv / predictions.json from inference
        ├── metrics.json         # test-set metrics + published-baseline comparison
        ├── predictions.csv      # per-sample test-set predictions
        ├── summary.md           # evaluation summary
        ├── metadata.json        # run provenance (PRD §17)
        └── report.md            # automated experiment report (PRD §100)
```

### Run IDs

A run ID is a single, filesystem-safe path segment. It comes from, in order of
precedence:

1. the `--run-id` command-line flag,
2. `experiment.run_id` in `configs/experiment.yaml`,
3. a generated `"<experiment.name>_<UTC timestamp>"`, e.g.
   `cgcnn_mp_e_form_20261010T153000Z`.

Pin the ID (flag or config) to write a reproduced run into a named directory;
leave it unset for a fresh, collision-free directory per run.

The root (`experiments/`) and the runs subdirectory (`runs/`) are configurable
through `experiment.experiments_dir` and `experiment.runs_dirname`.

## Producing a run

```bash
# 1. Train. Creates experiments/runs/<run_id>/ with checkpoints/ and logs/.
python src/cli/train_cli.py --run-id my_run

# 2. Optional: explanations and inference predictions for the same run.
python src/cli/explain_cli.py \
    --checkpoint experiments/runs/my_run/checkpoints/best.ckpt \
    --output-dir experiments/runs/my_run/explanations
python src/cli/predict_cli.py predict-batch \
    --checkpoint experiments/runs/my_run/checkpoints/best.ckpt \
    --input-dir path/to/cifs \
    --output-dir experiments/runs/my_run/predictions

# 3. Evaluate, plot, collect provenance and render report.md.
python src/cli/report_cli.py \
    --run-id my_run \
    --checkpoint experiments/runs/my_run/checkpoints/best.ckpt

# Re-render report.md from artifacts already on disk, without re-evaluating.
python src/cli/report_cli.py --run-id my_run --report-only
```

`report_cli.py` is the Phase 10 orchestrator and the only entry point that
enforces this layout. `evaluate_cli.py` remains the standalone Phase 7 tool:
it writes a flat directory of evaluation artifacts wherever `--output-dir`
points and is unaffected by the run layout.

## Legacy directories

These predate the layout above and were produced by earlier phases. They are
**retained permanently** under the policy above; nothing reads them except
when they are passed explicitly to `--log-dir` or `--checkpoint`.

They are deliberately not migrated into `runs/`. Folding them in would mean
inventing a run ID, a seed and a provenance record that were never captured,
which would misrepresent them as reproducible runs. Instead, a legacy run can
be *referenced* from a new run directory with `--copy-logs` and
`--copy-checkpoint` (see below), leaving the originals untouched.

| Directory | Origin | Notes |
| --- | --- | --- |
| `cgcnn_logs/` | Phase 6 `TensorBoardLogger("experiments", name="cgcnn_logs")` | `version_0` … `version_12` from repeated runs; no run IDs, so individual runs cannot be told apart. |
| `cgcnn_csv_logs/` | Phase 6 `CSVLogger("experiments", name="cgcnn_csv_logs")` | `version_0/metrics.csv`. No `lr` column: the learning rate was not logged before Phase 10 added `LearningRateMonitor`, so `learning_rate.png` cannot be generated from it. |
| `cgcnn_full_history/` | Manual concatenation of CSV logs | Merged history of the 100-epoch run. |
| `checkpoints/` | Phase 6 `ModelCheckpoint(dirpath="experiments/checkpoints")` | Checkpoints from several runs mixed together; `cgcnn-epoch=99-val_loss=0.0070.ckpt` is the best one. |
| `plots/` | `plot_metrics.py` (local script) | Ad-hoc `mae_plot.png`, `r2_plot.png`, `rmse_plot.png`. |

To report on a legacy run through the canonical layout, pass its artifacts to
`report_cli.py` explicitly. The flags *copy* the artifacts into the new run
directory; the originals stay where they are.

```bash
python src/cli/report_cli.py \
    --run-id legacy_100_epoch \
    --checkpoint "experiments/checkpoints/cgcnn-epoch=99-val_loss=0.0070.ckpt" \
    --log-dir experiments/cgcnn_full_history/metrics.csv \
    --copy-logs --copy-checkpoint --num-workers 0
```

This is how `runs/legacy_100_epoch/` was produced: 13,276 test graphs, MAE
0.0492 eV/atom, RMSE 0.0881, R² 0.9943. Its report has no learning-rate
figure, because the legacy log has no `lr` column.
