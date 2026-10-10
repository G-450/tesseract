"""Unit and integration tests for the Phase 9 inference subsystem.

Tests cover:
  - Exception hierarchy and attributes
  - PredictionResult / BatchPredictionResult dataclasses
  - Training/inference preprocessing parity (PRD 90, critical)
  - Model loading, structure loading and failure isolation
  - Batch prediction over a directory, with per-structure failure isolation
  - CSV / JSON export
  - predict_cli.py integration for both subcommands
"""

import csv
import json
import os
from typing import Any, List, Tuple

import pytest
import pytorch_lightning as pl
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from src.config.dataset_config import load_dataset_config
from src.config.model_config import CGCNNConfig
from src.graphs.builder import CrystalGraphBuilder
from src.inference.exceptions import (
    InferenceError,
    ModelLoadError,
    PredictionError,
    PredictionExportError,
    StructureLoadError,
)
from src.inference.exporter import CSV_COLUMNS, PredictionExporter
from src.inference.predictor import (
    BatchPredictionResult,
    PredictionResult,
    Predictor,
)
from src.models.lightning_module import CGCNNLightningModule
from src.utils.checkpoints import checkpoint_safe_globals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structure() -> Structure:
    """Returns a rock-salt NaCl structure."""
    lattice = Lattice.cubic(5.64)
    return Structure(
        lattice,
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _make_second_structure() -> Structure:
    """Returns a body-centred cubic iron structure."""
    lattice = Lattice.cubic(2.87)
    return Structure(lattice, ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def _write_cif(structure: Structure, path: str) -> str:
    """Writes a structure to a CIF file and returns the path."""
    structure.to(filename=path, fmt="cif")
    return path


def _make_model() -> CGCNNLightningModule:
    """Returns a small CGCNN model for testing."""
    config = CGCNNConfig()
    config.model.hidden_dim = 16
    config.model.num_conv_layers = 2
    model = CGCNNLightningModule(config)
    model.eval()
    return model


def _write_checkpoint(tmp_path: Any) -> Tuple[str, CGCNNLightningModule]:
    """Writes a Lightning checkpoint and returns its path and source model.

    The model uses the default CGCNNConfig so that the checkpoint loads
    against the project's model configuration.
    """
    model = CGCNNLightningModule(CGCNNConfig())
    model.eval()
    path = str(tmp_path / "trained.ckpt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"config": CGCNNConfig()},
            "pytorch-lightning_version": pl.__version__,
            "epoch": 0,
            "global_step": 0,
        },
        path,
    )
    return path, model


def _make_predictor(tmp_path: Any) -> Predictor:
    """Returns a Predictor with an in-memory model, bypassing checkpoints."""
    predictor = Predictor(
        checkpoint_path=str(tmp_path / "missing.ckpt"),
        device="cpu",
    )
    predictor._model = _make_model()
    return predictor


def _make_prediction(sample_id: str = "NaCl", energy: float = -2.0) -> PredictionResult:
    """Returns a PredictionResult for export tests."""
    return PredictionResult(
        sample_id=sample_id,
        formation_energy=energy,
        formula="NaCl",
        num_atoms=2,
        num_edges=24,
        source_path=f"{sample_id}.cif",
        metadata={"units": "eV/atom"},
    )


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_all_inference_errors_inherit_from_inference_error() -> None:
    """Every inference exception descends from InferenceError."""
    for error in (
        ModelLoadError,
        StructureLoadError,
        PredictionError,
        PredictionExportError,
    ):
        assert issubclass(error, InferenceError)
    assert issubclass(InferenceError, Exception)


def test_model_load_error_attributes() -> None:
    """ModelLoadError records the checkpoint path and reason."""
    error = ModelLoadError("best.ckpt", "corrupt file")
    assert error.checkpoint_path == "best.ckpt"
    assert error.reason == "corrupt file"
    assert "best.ckpt" in str(error)
    assert "corrupt file" in str(error)


def test_structure_load_error_attributes() -> None:
    """StructureLoadError records the path and reason."""
    error = StructureLoadError("bad.cif", "unparseable")
    assert error.path == "bad.cif"
    assert error.reason == "unparseable"
    assert "bad.cif" in str(error)


def test_prediction_error_attributes() -> None:
    """PredictionError records the sample id and reason."""
    error = PredictionError("NaCl", "forward pass failed")
    assert error.sample_id == "NaCl"
    assert error.reason == "forward pass failed"
    assert "NaCl" in str(error)


def test_prediction_export_error_attributes() -> None:
    """PredictionExportError records the path and reason."""
    error = PredictionExportError("out.csv", "permission denied")
    assert error.path == "out.csv"
    assert error.reason == "permission denied"
    assert "out.csv" in str(error)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


def test_prediction_result_to_dict_is_json_serialisable() -> None:
    """to_dict() exposes the documented keys and survives json.dumps."""
    payload = _make_prediction().to_dict()
    assert payload["sample_id"] == "NaCl"
    assert payload["formation_energy_ev_per_atom"] == -2.0
    assert payload["num_atoms"] == 2
    assert payload["num_edges"] == 24
    json.dumps(payload)


def test_batch_result_summary_on_empty_run() -> None:
    """An empty batch reports zero counts and no energy statistics."""
    summary = BatchPredictionResult().summary()
    assert summary["num_attempted"] == 0
    assert summary["num_succeeded"] == 0
    assert summary["num_failed"] == 0
    assert "mean_formation_energy_ev_per_atom" not in summary


def test_batch_result_summary_with_single_prediction() -> None:
    """A one-structure batch reports a zero standard deviation."""
    result = BatchPredictionResult(predictions=[_make_prediction(energy=-1.5)])
    summary = result.summary()
    assert summary["num_succeeded"] == 1
    assert summary["mean_formation_energy_ev_per_atom"] == pytest.approx(-1.5)
    assert summary["std_formation_energy_ev_per_atom"] == 0.0


def test_batch_result_summary_counts_failures() -> None:
    """Failures are counted in num_attempted but not in the statistics."""
    result = BatchPredictionResult(
        predictions=[
            _make_prediction("a", -1.0),
            _make_prediction("b", -3.0),
        ],
        failures=[("c", "unparseable")],
    )
    summary = result.summary()
    assert result.num_attempted == 3
    assert summary["num_succeeded"] == 2
    assert summary["num_failed"] == 1
    assert summary["mean_formation_energy_ev_per_atom"] == pytest.approx(-2.0)
    assert summary["min_formation_energy_ev_per_atom"] == pytest.approx(-3.0)
    assert summary["max_formation_energy_ev_per_atom"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Preprocessing parity (PRD 90) -- a divergence here is a critical defect
# ---------------------------------------------------------------------------


def test_inference_graph_is_identical_to_training_graph(tmp_path: Any) -> None:
    """The same structure yields identical tensors in training and inference.

    The training pipeline builds graphs with CrystalGraphBuilder and the
    dataset config's GraphConfig; inference must produce bit-identical
    tensors for the same structure (PRD 90).
    """
    structure = _make_structure()
    config = load_dataset_config("configs/dataset.yaml")

    training_graph = CrystalGraphBuilder(config.graph).build(
        structure=structure,
        record_id=7,
        fingerprint="training-fingerprint",
        target=-1.23,
    )
    inference_graph = _make_predictor(tmp_path).build_graph(structure)

    assert torch.equal(inference_graph.x, training_graph.x)
    assert torch.equal(inference_graph.edge_index, training_graph.edge_index)
    assert torch.equal(inference_graph.edge_attr, training_graph.edge_attr)
    assert torch.equal(inference_graph.edge_image, training_graph.edge_image)
    assert inference_graph.num_nodes == training_graph.num_nodes
    assert inference_graph.num_edges == training_graph.num_edges
    assert inference_graph.x.dtype == training_graph.x.dtype
    assert inference_graph.edge_attr.dtype == training_graph.edge_attr.dtype


def test_predictor_uses_graph_config_from_dataset_config(tmp_path: Any) -> None:
    """The predictor's graph settings come from the dataset configuration."""
    predictor = _make_predictor(tmp_path)
    expected = load_dataset_config("configs/dataset.yaml").graph
    assert predictor.graph_config == expected
    assert predictor.builder.config == expected


def test_build_graph_uses_sentinel_target_and_record_id(tmp_path: Any) -> None:
    """Inference graphs carry sentinel provenance, since no target is known."""
    graph = _make_predictor(tmp_path).build_graph(_make_structure())
    assert graph.y.item() == pytest.approx(0.0)
    assert graph.dataset_record_id == -1
    assert graph.dataset_fingerprint == "inference"


# ---------------------------------------------------------------------------
# Model and structure loading
# ---------------------------------------------------------------------------


def test_load_model_raises_for_missing_checkpoint(tmp_path: Any) -> None:
    """A missing checkpoint raises ModelLoadError rather than OSError."""
    predictor = Predictor(checkpoint_path=str(tmp_path / "absent.ckpt"), device="cpu")
    with pytest.raises(ModelLoadError) as excinfo:
        predictor.load_model()
    assert "does not exist" in str(excinfo.value)


def test_load_model_raises_for_corrupt_checkpoint(tmp_path: Any) -> None:
    """A checkpoint that cannot be deserialised raises ModelLoadError."""
    path = tmp_path / "corrupt.ckpt"
    path.write_text("not a checkpoint", encoding="utf-8")
    predictor = Predictor(checkpoint_path=str(path), device="cpu")
    with pytest.raises(ModelLoadError):
        predictor.load_model()


def test_load_model_is_cached(tmp_path: Any) -> None:
    """The model is loaded once and returned from cache afterwards."""
    predictor = _make_predictor(tmp_path)
    first = predictor.load_model()
    assert predictor.load_model() is first


def test_load_model_returns_frozen_eval_model(tmp_path: Any) -> None:
    """A checkpoint loads in eval mode with every parameter frozen (PRD 89)."""
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    predictor = Predictor(checkpoint_path=checkpoint_path, device="cpu")
    model = predictor.load_model()

    assert not model.training
    assert all(not param.requires_grad for param in model.parameters())


def test_checkpoint_loads_when_weights_only_is_enforced(tmp_path: Any) -> None:
    """Project config classes are allowlisted for weights_only loading.

    From PyTorch 2.6 onwards torch.load defaults to weights_only=True, which
    rejects the pickled CGCNNConfig stored in a checkpoint's hyperparameters.
    This asserts the allowlist covers it, independently of the local default.
    """
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    with checkpoint_safe_globals():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint["hyper_parameters"]["config"], CGCNNConfig)


def test_weights_only_load_is_restricted_without_the_allowlist(tmp_path: Any) -> None:
    """Outside the allowlist, weights_only loading still blocks the globals.

    Guards that the test above is meaningful and that the allowlist is scoped
    to the load rather than applied process-wide.
    """
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    with pytest.raises(Exception):
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def test_load_model_survives_enforced_weights_only(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """load_model() works when torch.load enforces weights_only.

    Reproduces the PyTorch >= 2.6 default through Lightning's own loader, so
    the allowlist is exercised on the real checkpoint-loading path regardless
    of the installed PyTorch version.
    """
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    original_load = torch.load

    def enforced_load(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", enforced_load)

    predictor = Predictor(checkpoint_path=checkpoint_path, device="cpu")
    model = predictor.load_model()
    assert not model.training


def test_prediction_matches_the_checkpointed_model(tmp_path: Any) -> None:
    """Predictions reproduce the checkpointed model's own forward pass.

    This is the Phase 9 acceptance criterion that predictions must match the
    trained model: the predictor's output is compared against the in-memory
    model the checkpoint was saved from.
    """
    checkpoint_path, source_model = _write_checkpoint(tmp_path)
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        model_config_path=str(tmp_path / "absent-model.yaml"),
        device="cpu",
    )

    structure = _make_structure()
    graph = predictor.build_graph(structure)
    with torch.no_grad():
        expected = source_model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            torch.zeros(graph.num_nodes, dtype=torch.long),
        ).view(-1)[0]

    result = predictor.predict_structure(structure)
    assert result.formation_energy == pytest.approx(float(expected), abs=1e-6)


def test_load_structure_raises_for_missing_file(tmp_path: Any) -> None:
    """A missing structure file raises StructureLoadError."""
    with pytest.raises(StructureLoadError):
        Predictor.load_structure(str(tmp_path / "absent.cif"))


def test_load_structure_raises_for_invalid_cif(tmp_path: Any) -> None:
    """A file that is not valid CIF raises StructureLoadError."""
    path = tmp_path / "broken.cif"
    path.write_text("this is not a cif file", encoding="utf-8")
    with pytest.raises(StructureLoadError):
        Predictor.load_structure(str(path))


def test_load_structure_parses_valid_cif(tmp_path: Any) -> None:
    """A valid CIF round-trips into an equivalent pymatgen Structure."""
    original = _make_structure()
    path = _write_cif(original, str(tmp_path / "NaCl.cif"))
    loaded = Predictor.load_structure(path)
    assert len(loaded) == len(original)
    assert loaded.composition.reduced_formula == original.composition.reduced_formula


# ---------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------


def test_discover_structures_sorts_and_filters(tmp_path: Any) -> None:
    """Discovery returns only supported files, sorted by name."""
    directory = tmp_path / "structures"
    directory.mkdir()
    _write_cif(_make_structure(), str(directory / "b_NaCl.cif"))
    _write_cif(_make_second_structure(), str(directory / "a_Fe.cif"))
    (directory / "notes.txt").write_text("ignore me", encoding="utf-8")

    found = Predictor.discover_structures(str(directory))
    assert [os.path.basename(path) for path in found] == ["a_Fe.cif", "b_NaCl.cif"]


def test_discover_structures_raises_for_missing_directory(tmp_path: Any) -> None:
    """A missing directory raises StructureLoadError."""
    with pytest.raises(StructureLoadError):
        Predictor.discover_structures(str(tmp_path / "absent"))


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


def test_predict_structure_returns_finite_energy(tmp_path: Any) -> None:
    """Predicting a structure yields a finite scalar and correct graph counts."""
    predictor = _make_predictor(tmp_path)
    structure = _make_structure()
    result = predictor.predict_structure(structure)

    expected_graph = predictor.build_graph(structure)
    assert isinstance(result.formation_energy, float)
    assert torch.isfinite(torch.tensor(result.formation_energy))
    assert result.sample_id == "NaCl"
    assert result.formula == "NaCl"
    assert result.num_atoms == expected_graph.num_nodes
    assert result.num_edges == expected_graph.num_edges
    assert result.metadata["units"] == "eV/atom"
    assert result.metadata["device"] == "cpu"


def test_predict_structure_matches_direct_forward_pass(tmp_path: Any) -> None:
    """The predictor's output equals a manual forward pass on the same graph."""
    predictor = _make_predictor(tmp_path)
    structure = _make_structure()
    graph = predictor.build_graph(structure)

    model = predictor.load_model()
    with torch.no_grad():
        expected = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            torch.zeros(graph.num_nodes, dtype=torch.long),
        ).view(-1)[0]

    result = predictor.predict_structure(structure)
    assert result.formation_energy == pytest.approx(float(expected), abs=1e-5)


def test_predict_structure_wraps_failures_in_prediction_error(tmp_path: Any) -> None:
    """A forward-pass failure surfaces as PredictionError with the sample id."""
    predictor = _make_predictor(tmp_path)

    def _boom(_graphs: Any) -> torch.Tensor:
        raise RuntimeError("device mismatch")

    predictor._forward = _boom  # type: ignore[assignment]
    with pytest.raises(PredictionError) as excinfo:
        predictor.predict_structure(_make_structure(), sample_id="NaCl")
    assert excinfo.value.sample_id == "NaCl"
    assert "device mismatch" in str(excinfo.value)


def test_predict_cif_defaults_sample_id_to_file_stem(tmp_path: Any) -> None:
    """predict_cif names the result after the file and records its path."""
    path = _write_cif(_make_structure(), str(tmp_path / "rocksalt.cif"))
    result = _make_predictor(tmp_path).predict_cif(path)
    assert result.sample_id == "rocksalt"
    assert result.source_path == path


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


def test_predict_batch_predicts_every_structure(tmp_path: Any) -> None:
    """A clean directory produces one prediction per structure, in order."""
    directory = tmp_path / "structures"
    directory.mkdir()
    _write_cif(_make_structure(), str(directory / "a_NaCl.cif"))
    _write_cif(_make_second_structure(), str(directory / "b_Fe.cif"))

    result = _make_predictor(tmp_path).predict_batch(str(directory))
    assert result.num_attempted == 2
    assert not result.failures
    assert [p.sample_id for p in result.predictions] == ["a_NaCl", "b_Fe"]
    assert all(
        torch.isfinite(torch.tensor(p.formation_energy)) for p in result.predictions
    )


def test_predict_batch_isolates_unparseable_structures(tmp_path: Any) -> None:
    """A corrupt CIF is logged and skipped without aborting the batch."""
    directory = tmp_path / "structures"
    directory.mkdir()
    _write_cif(_make_structure(), str(directory / "good.cif"))
    (directory / "broken.cif").write_text("not a cif", encoding="utf-8")

    result = _make_predictor(tmp_path).predict_batch(str(directory))
    assert len(result.predictions) == 1
    assert result.predictions[0].sample_id == "good"
    assert [sample_id for sample_id, _ in result.failures] == ["broken"]
    assert result.num_attempted == 2


def test_predict_batch_accepts_explicit_path_sequence(tmp_path: Any) -> None:
    """A list of paths is predicted in the given order."""
    first = _write_cif(_make_structure(), str(tmp_path / "one.cif"))
    second = _write_cif(_make_second_structure(), str(tmp_path / "two.cif"))
    result = _make_predictor(tmp_path).predict_batch([second, first])
    assert [p.sample_id for p in result.predictions] == ["two", "one"]


def test_predict_batch_returns_empty_result_for_empty_directory(tmp_path: Any) -> None:
    """An empty directory yields no predictions and no failures."""
    directory = tmp_path / "empty"
    directory.mkdir()
    result = _make_predictor(tmp_path).predict_batch(str(directory))
    assert result.num_attempted == 0
    assert result.summary()["num_attempted"] == 0


def test_predict_batch_respects_batch_size(tmp_path: Any) -> None:
    """Structures are scored in chunks no larger than the batch size."""
    directory = tmp_path / "structures"
    directory.mkdir()
    for index in range(5):
        _write_cif(_make_structure(), str(directory / f"s{index}.cif"))

    predictor = _make_predictor(tmp_path)
    original = predictor._forward
    chunk_sizes: List[int] = []

    def _record(graphs: Any) -> torch.Tensor:
        chunk_sizes.append(len(graphs))
        return original(graphs)

    predictor._forward = _record  # type: ignore[method-assign]
    result = predictor.predict_batch(str(directory), batch_size=2)

    assert len(result.predictions) == 5
    assert chunk_sizes == [2, 2, 1]


def test_predict_batch_falls_back_to_single_structures(tmp_path: Any) -> None:
    """When a batched forward pass fails, structures are retried one by one."""
    directory = tmp_path / "structures"
    directory.mkdir()
    _write_cif(_make_structure(), str(directory / "a.cif"))
    _write_cif(_make_second_structure(), str(directory / "b.cif"))

    predictor = _make_predictor(tmp_path)
    original = predictor._forward

    def _fail_on_batches(graphs: Any) -> torch.Tensor:
        if len(graphs) > 1:
            raise RuntimeError("batched kernel unavailable")
        return original(graphs)

    predictor._forward = _fail_on_batches  # type: ignore[method-assign]
    result = predictor.predict_batch(str(directory), batch_size=8)

    assert len(result.predictions) == 2
    assert not result.failures


def test_batched_and_single_predictions_agree(tmp_path: Any) -> None:
    """Batched scoring matches per-structure scoring for the same graphs."""
    directory = tmp_path / "structures"
    directory.mkdir()
    _write_cif(_make_structure(), str(directory / "a.cif"))
    _write_cif(_make_second_structure(), str(directory / "b.cif"))

    predictor = _make_predictor(tmp_path)
    batched = predictor.predict_batch(str(directory), batch_size=8)
    single = predictor.predict_batch(str(directory), batch_size=1)

    for batched_result, single_result in zip(batched.predictions, single.predictions):
        assert batched_result.sample_id == single_result.sample_id
        assert batched_result.formation_energy == pytest.approx(
            single_result.formation_energy, abs=1e-5
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_to_csv_writes_documented_columns(tmp_path: Any) -> None:
    """The CSV carries the documented header and one row per prediction."""
    path = str(tmp_path / "predictions.csv")
    PredictionExporter.to_csv(
        [_make_prediction("a", -1.0), _make_prediction("b")], path
    )

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0].keys()) == list(CSV_COLUMNS)
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert float(rows[0]["formation_energy_ev_per_atom"]) == pytest.approx(-1.0)


def test_to_json_includes_summary_and_failures(tmp_path: Any) -> None:
    """The JSON payload carries predictions, summary and failures."""
    path = str(tmp_path / "predictions.json")
    result = BatchPredictionResult(
        predictions=[_make_prediction()],
        failures=[("broken", "unparseable")],
    )
    PredictionExporter.to_json(
        result.predictions,
        path,
        summary=result.summary(),
        failures=result.failures,
    )

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["predictions"][0]["sample_id"] == "NaCl"
    assert payload["summary"]["num_failed"] == 1
    assert payload["failures"] == [{"sample_id": "broken", "reason": "unparseable"}]


def test_export_creates_both_artifacts_and_parent_directory(tmp_path: Any) -> None:
    """export() creates the output directory and writes CSV and JSON."""
    output_dir = str(tmp_path / "nested" / "predictions")
    result = BatchPredictionResult(predictions=[_make_prediction()])
    artifacts = PredictionExporter.export(result, output_dir)

    assert [os.path.basename(path) for path in artifacts] == [
        "predictions.csv",
        "predictions.json",
    ]
    assert all(os.path.exists(path) for path in artifacts)


def test_export_single_respects_extension(tmp_path: Any) -> None:
    """export_single() selects the format from the file extension."""
    prediction = _make_prediction()
    csv_path = PredictionExporter.export_single(prediction, str(tmp_path / "one.csv"))
    json_path = PredictionExporter.export_single(prediction, str(tmp_path / "one.json"))
    assert os.path.exists(csv_path)
    with open(json_path, "r", encoding="utf-8") as handle:
        assert json.load(handle)["predictions"][0]["sample_id"] == "NaCl"


def test_export_single_rejects_unsupported_extension(tmp_path: Any) -> None:
    """An unsupported extension raises PredictionExportError."""
    with pytest.raises(PredictionExportError):
        PredictionExporter.export_single(_make_prediction(), str(tmp_path / "x.txt"))


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_predict_cli_single_structure(tmp_path: Any, monkeypatch: Any) -> None:
    """predict_cli.py 'predict' writes a JSON export for one structure."""
    from src.cli.predict_cli import main

    monkeypatch.setattr(
        "src.inference.predictor.Predictor.load_model",
        lambda self: _make_model(),
    )

    cif_path = _write_cif(_make_structure(), str(tmp_path / "NaCl.cif"))
    output_path = str(tmp_path / "out" / "NaCl.json")
    checkpoint = tmp_path / "dummy.ckpt"
    checkpoint.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict_cli.py",
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--input",
            cif_path,
            "--output",
            output_path,
            "--device",
            "cpu",
        ],
    )
    main()

    with open(output_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["predictions"][0]["sample_id"] == "NaCl"


def test_predict_cli_batch(tmp_path: Any, monkeypatch: Any) -> None:
    """predict_cli.py 'predict-batch' exports CSV and JSON for a directory."""
    from src.cli.predict_cli import main

    monkeypatch.setattr(
        "src.inference.predictor.Predictor.load_model",
        lambda self: _make_model(),
    )

    directory = tmp_path / "structures"
    directory.mkdir()
    _write_cif(_make_structure(), str(directory / "NaCl.cif"))
    _write_cif(_make_second_structure(), str(directory / "Fe.cif"))
    (directory / "broken.cif").write_text("not a cif", encoding="utf-8")

    output_dir = str(tmp_path / "out")
    checkpoint = tmp_path / "dummy.ckpt"
    checkpoint.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict_cli.py",
            "predict-batch",
            "--checkpoint",
            str(checkpoint),
            "--input-dir",
            str(directory),
            "--output-dir",
            output_dir,
            "--device",
            "cpu",
        ],
    )
    main()

    with open(os.path.join(output_dir, "predictions.json"), "r", encoding="utf-8") as h:
        payload = json.load(h)
    assert payload["summary"]["num_succeeded"] == 2
    assert payload["summary"]["num_failed"] == 1
    assert os.path.exists(os.path.join(output_dir, "predictions.csv"))


def test_predict_cli_exits_on_missing_checkpoint(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A missing checkpoint exits non-zero instead of raising a traceback."""
    from src.cli.predict_cli import main

    cif_path = _write_cif(_make_structure(), str(tmp_path / "NaCl.cif"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict_cli.py",
            "predict",
            "--checkpoint",
            str(tmp_path / "absent.ckpt"),
            "--input",
            cif_path,
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
