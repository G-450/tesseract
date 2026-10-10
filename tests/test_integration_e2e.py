"""End-to-end integration tests (PRD §111).

These tests exercise the full pipeline on a small synthetic fixture:
graph construction -> model init -> 1 training epoch -> evaluation ->
prediction -> report rendering.  They verify that the system functions
as an integrated whole, not just individual subsystems.
"""

import json
import os
from typing import Any, Dict

import pytest
import torch
from torch_geometric.data import Data

from src.config.dataset_config import DataLoaderConfig, DatasetConfig
from src.config.model_config import CGCNNConfig, ModelConfig, OptimizerConfig
from src.data.datamodule import CrystalDataModule
from src.evaluation.evaluator import EvaluationResult
from src.evaluation.metrics import compute_all_metrics
from src.models.lightning_module import CGCNNLightningModule
from src.utils.run_layout import RunLayout

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NODE_DIM = 92
EDGE_DIM = 41
NUM_GRAPHS = 20


def _make_graph(num_atoms: int = 4, seed: int = 0) -> Data:
    """Creates a single synthetic crystal graph with the right shape."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randint(0, NODE_DIM, (num_atoms,), generator=rng)
    num_edges = num_atoms * 3
    edge_index = torch.stack(
        [
            torch.randint(0, num_atoms, (num_edges,), generator=rng),
            torch.randint(0, num_atoms, (num_edges,), generator=rng),
        ]
    )
    edge_attr = torch.randn(num_edges, EDGE_DIM, generator=rng)
    y = torch.randn(1, generator=rng)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _write_shard_dataset(processed_dir: str, n: int = NUM_GRAPHS) -> None:
    """Writes a synthetic sharded dataset to disk."""
    graphs_dir = os.path.join(processed_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    graphs = [_make_graph(num_atoms=3 + i % 5, seed=i) for i in range(n)]
    shard_name = "shard_00000.pt"
    torch.save(graphs, os.path.join(graphs_dir, shard_name))

    manifest = {
        "fingerprint": "e2e_test",
        "total_records_processed": n,
        "successful_graphs": n,
        "shard_lengths": {shard_name: n},
    }
    with open(os.path.join(graphs_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


@pytest.fixture
def e2e_dir(tmp_path: Any) -> Dict[str, str]:
    """Sets up the full directory tree for an e2e test."""
    processed = str(tmp_path / "processed")
    _write_shard_dataset(processed, NUM_GRAPHS)
    return {
        "processed": processed,
        "run_root": str(tmp_path / "runs"),
        "tmp": str(tmp_path),
    }


@pytest.fixture
def dataset_config(e2e_dir: Dict[str, str]) -> DatasetConfig:
    return DatasetConfig(
        processed_data_dir=e2e_dir["processed"],
        dataloader=DataLoaderConfig(
            batch_size=4,
            num_workers=0,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
        ),
    )


@pytest.fixture
def model_config() -> CGCNNConfig:
    return CGCNNConfig(
        model=ModelConfig(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            hidden_dim=16,
            num_conv_layers=1,
        ),
        optimizer=OptimizerConfig(lr=0.01),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full-pipeline integration test (PRD §111)."""

    def test_train_evaluate_predict_report(
        self,
        e2e_dir: Dict[str, str],
        dataset_config: DatasetConfig,
        model_config: CGCNNConfig,
    ) -> None:
        """Runs the pipeline: train 1 epoch -> evaluate -> predict -> report."""
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint

        seed = 42
        pl.seed_everything(seed, workers=True)

        # ── 1. Train for 1 epoch ────────────────────────────────────────
        datamodule = CrystalDataModule(config=dataset_config, seed=seed)
        model = CGCNNLightningModule(config=model_config)

        run_id = "e2e_test"
        layout = RunLayout.for_run(e2e_dir["run_root"], run_id).create()

        checkpoint_cb = ModelCheckpoint(
            dirpath=layout.checkpoints_dir,
            filename="best",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            callbacks=[checkpoint_cb],
            accelerator="cpu",
        )
        trainer.fit(model, datamodule)

        best_ckpt = checkpoint_cb.best_model_path
        assert os.path.isfile(best_ckpt), "Checkpoint was not saved"

        # ── 2. Evaluate on test split ───────────────────────────────────
        datamodule_eval = CrystalDataModule(config=dataset_config, seed=seed)
        datamodule_eval.setup(stage="test")
        test_loader = datamodule_eval.test_dataloader()

        from src.utils.checkpoints import checkpoint_safe_globals

        with checkpoint_safe_globals():
            eval_model = CGCNNLightningModule.load_from_checkpoint(
                best_ckpt, config=model_config
            )
        eval_model.eval()
        eval_model.freeze()
        eval_model.to("cpu")

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                out = eval_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                all_preds.append(out.view(-1).cpu())
                all_targets.append(batch.y.float().view(-1).cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        assert len(preds) > 0, "No test predictions generated"

        metrics = compute_all_metrics(preds, targets)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0.0

        result = EvaluationResult(
            predictions=preds,
            targets=targets,
            metrics=metrics,
        )

        # ── 3. Export metrics and report ────────────────────────────────
        metrics_path = layout.metrics_json
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        assert os.path.isfile(metrics_path)

        # Generate plots
        from src.visualization.plots import generate_all_plots

        plots = generate_all_plots(result=result, output_dir=layout.plots_dir)
        assert "parity_plot" in plots
        assert "residual_plot" in plots

        # Render experiment report
        from src.evaluation.experiment_report import ExperimentReporter
        from src.utils.provenance import RunProvenance

        provenance = RunProvenance(
            run_id=run_id,
            experiment_name="e2e_test",
            seed=seed,
            generated_at="2026-10-11T00:00:00Z",
        )
        reporter = ExperimentReporter(baseline_mae=0.039)
        report_path = reporter.export(
            layout=layout,
            provenance=provenance,
            metrics=metrics,
        )
        assert os.path.isfile(report_path)

        with open(report_path, "r") as f:
            report_text = f.read()
        assert "MAE" in report_text
        assert run_id in report_text

        # ── 4. Verify prediction from checkpoint ────────────────────────
        single_graph = _make_graph(num_atoms=5, seed=999)
        batch_idx = torch.zeros(single_graph.x.shape[0], dtype=torch.long)
        with torch.no_grad():
            single_pred = eval_model(
                single_graph.x,
                single_graph.edge_index,
                single_graph.edge_attr,
                batch_idx,
            )
        assert (
            single_pred.numel() == 1
        ), f"Expected scalar prediction, got {single_pred.shape}"


class TestConfigValidation:
    """Config validation rejects unknown keys (PRD §122)."""

    def test_dataset_config_rejects_unknown_keys(self, tmp_path: Any) -> None:
        import yaml

        from src.config.dataset_config import load_dataset_config
        from src.config.exceptions import UnknownConfigKeyError

        cfg = tmp_path / "bad_dataset.yaml"
        cfg.write_text(yaml.dump({"dataset": {"raw_data_path": "x", "bogus_key": 123}}))
        with pytest.raises(UnknownConfigKeyError):
            load_dataset_config(str(cfg))

    def test_model_config_rejects_unknown_keys(self, tmp_path: Any) -> None:
        import yaml

        from src.config.exceptions import UnknownConfigKeyError
        from src.config.model_config import load_cgcnn_config

        cfg = tmp_path / "bad_model.yaml"
        cfg.write_text(yaml.dump({"model": {"hidden_dim": 64, "bogus": True}}))
        with pytest.raises(UnknownConfigKeyError):
            load_cgcnn_config(str(cfg))

    def test_dataset_config_rejects_unknown_top_level(self, tmp_path: Any) -> None:
        import yaml

        from src.config.dataset_config import load_dataset_config
        from src.config.exceptions import UnknownConfigKeyError

        cfg = tmp_path / "bad_top.yaml"
        cfg.write_text(yaml.dump({"dataset": {}, "extra_section": {}}))
        with pytest.raises(UnknownConfigKeyError):
            load_dataset_config(str(cfg))

    def test_model_config_rejects_unknown_top_level(self, tmp_path: Any) -> None:
        import yaml

        from src.config.exceptions import UnknownConfigKeyError
        from src.config.model_config import load_cgcnn_config

        cfg = tmp_path / "bad_top.yaml"
        cfg.write_text(yaml.dump({"model": {}, "wrong_section": {}}))
        with pytest.raises(UnknownConfigKeyError):
            load_cgcnn_config(str(cfg))

    def test_dataset_config_accepts_valid(self, tmp_path: Any) -> None:
        import yaml

        from src.config.dataset_config import load_dataset_config

        cfg = tmp_path / "good.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "dataset": {
                        "raw_data_path": "data.json",
                        "graph": {"cutoff_radius": 6.0},
                        "dataloader": {"batch_size": 16},
                    }
                }
            )
        )
        config = load_dataset_config(str(cfg))
        assert config.raw_data_path == "data.json"
        assert config.graph.cutoff_radius == 6.0
        assert config.dataloader.batch_size == 16

    def test_model_config_accepts_valid(self, tmp_path: Any) -> None:
        import yaml

        from src.config.model_config import load_cgcnn_config

        cfg = tmp_path / "good.yaml"
        cfg.write_text(
            yaml.dump({"model": {"hidden_dim": 128}, "optimizer": {"lr": 0.005}})
        )
        config = load_cgcnn_config(str(cfg))
        assert config.model.hidden_dim == 128
        assert config.optimizer.lr == 0.005


class TestSplitSeedDeterminism:
    """The split seed from experiment config controls the data split."""

    def test_different_seeds_give_different_splits(
        self, e2e_dir: Dict[str, str], dataset_config: DatasetConfig
    ) -> None:
        dm1 = CrystalDataModule(config=dataset_config, seed=1)
        dm1.setup()
        dm2 = CrystalDataModule(config=dataset_config, seed=2)
        dm2.setup()

        indices_1 = sorted(dm1.test_dataset.indices)  # type: ignore[union-attr]
        indices_2 = sorted(dm2.test_dataset.indices)  # type: ignore[union-attr]
        assert indices_1 != indices_2, "Different seeds should produce different splits"

    def test_same_seed_gives_same_split(
        self, e2e_dir: Dict[str, str], dataset_config: DatasetConfig
    ) -> None:
        dm1 = CrystalDataModule(config=dataset_config, seed=99)
        dm1.setup()
        dm2 = CrystalDataModule(config=dataset_config, seed=99)
        dm2.setup()

        indices_1 = sorted(dm1.test_dataset.indices)  # type: ignore[union-attr]
        indices_2 = sorted(dm2.test_dataset.indices)  # type: ignore[union-attr]
        assert indices_1 == indices_2


class TestUnifiedCLI:
    """Smoke tests for the unified CLI dispatcher."""

    def test_doctor_help(self) -> None:
        """doctor --help should not crash."""
        from src.cli.main_cli import _run_doctor

        with pytest.raises(SystemExit) as exc_info:
            _run_doctor(["--help"])
        assert exc_info.value.code == 0

    def test_clean_cache_dry_run(self, tmp_path: Any) -> None:
        """clean-cache --dry-run should list files without deleting."""
        import yaml

        processed = tmp_path / "proc"
        processed.mkdir()
        (processed / "shard_00000.pt").write_bytes(b"\x00")

        cfg_path = tmp_path / "ds.yaml"
        cfg_path.write_text(
            yaml.dump({"dataset": {"processed_data_dir": str(processed)}})
        )

        from src.cli.main_cli import _run_clean_cache

        _run_clean_cache(["--config", str(cfg_path), "--dry-run"])
        assert (processed / "shard_00000.pt").exists(), "dry-run should not delete"

    def test_clean_cache_deletes(self, tmp_path: Any) -> None:
        """clean-cache should delete .pt files."""
        import yaml

        processed = tmp_path / "proc"
        processed.mkdir()
        (processed / "shard_00000.pt").write_bytes(b"\x00")

        cfg_path = tmp_path / "ds.yaml"
        cfg_path.write_text(
            yaml.dump({"dataset": {"processed_data_dir": str(processed)}})
        )

        from src.cli.main_cli import _run_clean_cache

        _run_clean_cache(["--config", str(cfg_path)])
        assert not (processed / "shard_00000.pt").exists()
