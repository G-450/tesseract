"""Run provenance capture (PRD §17).

Reproducibility is a first-class feature: every experiment records the git
commit, timestamp, seed, configuration snapshot, library versions, hardware
information, CUDA version, dataset version and model version. This module
collects that record once, so every downstream artifact describes the same
run.

Collection never raises on a missing source. A machine without git, CUDA or a
dataset manifest still produces a complete provenance record with the
unavailable fields set to ``None``, because a report is more useful than an
aborted run.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TRACKED_LIBRARIES: Tuple[str, ...] = (
    "torch",
    "torch-geometric",
    "pytorch-lightning",
    "torchmetrics",
    "pymatgen",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "captum",
    "ijson",
    "pyyaml",
)
"""Distributions whose versions are recorded for every run."""

PROVENANCE_SCHEMA_VERSION = 1
"""Schema version of the emitted metadata document."""

_GIT_TIMEOUT_SECONDS = 10


@dataclass
class GitProvenance:
    """State of the working tree the run was launched from.

    Attributes:
        commit: Full commit hash, or ``None`` outside a git repository.
        branch: Current branch name, or ``None`` when detached or unavailable.
        dirty: Whether tracked files had uncommitted changes, or ``None``
            when it could not be determined.
    """

    commit: Optional[str] = None
    branch: Optional[str] = None
    dirty: Optional[bool] = None


@dataclass
class HardwareProvenance:
    """Hardware and platform the run executed on.

    Attributes:
        platform: Platform description string.
        machine: Machine architecture.
        processor: Processor identifier.
        python_version: Interpreter version.
        cpu_count: Number of logical CPUs, or ``None`` when unknown.
        cuda_available: Whether CUDA was visible to PyTorch.
        cuda_version: CUDA version PyTorch was built against, if any.
        cudnn_version: cuDNN version, if any.
        gpu_names: Names of the visible CUDA devices.
    """

    platform: str = ""
    machine: str = ""
    processor: str = ""
    python_version: str = ""
    cpu_count: Optional[int] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    gpu_names: List[str] = field(default_factory=list)


@dataclass
class DatasetProvenance:
    """Version and shape of the dataset the run consumed.

    Attributes:
        fingerprint: SHA-256 fingerprint of the raw dataset file.
        raw_data_path: Configured path of the raw dataset.
        processed_data_dir: Configured directory of the processed shards.
        target_column: Name of the regression target column.
        total_records: Records seen during preprocessing.
        valid_records: Records that passed validation.
        successful_graphs: Graphs written to the processed shards.
        failed_graphs: Structures that failed graph construction.
        shards_created: Number of serialized shards.
        graph_config: Graph construction settings recorded in the manifest.
    """

    fingerprint: Optional[str] = None
    raw_data_path: Optional[str] = None
    processed_data_dir: Optional[str] = None
    target_column: Optional[str] = None
    total_records: Optional[int] = None
    valid_records: Optional[int] = None
    successful_graphs: Optional[int] = None
    failed_graphs: Optional[int] = None
    shards_created: Optional[int] = None
    graph_config: Optional[Dict[str, Any]] = None


@dataclass
class RunProvenance:
    """Complete provenance record for one experiment run (PRD §17).

    Attributes:
        schema_version: Version of this document's schema.
        run_id: The run identifier.
        experiment_name: Configured experiment name.
        generated_at: UTC ISO-8601 timestamp of collection.
        seed: Global random seed for the run.
        checkpoint_path: Checkpoint the run was evaluated from.
        checkpoint_sha256: Digest of that checkpoint file, if readable.
        model_version: Identifier of the evaluated model artifact.
        git: Git state of the working tree.
        hardware: Hardware and platform description.
        dataset: Dataset version and shape.
        library_versions: Installed versions of the tracked libraries.
        config_snapshots: Mapping of snapshot name to its path in the run.
        configs: The configuration values the run used.
    """

    schema_version: int = PROVENANCE_SCHEMA_VERSION
    run_id: str = ""
    experiment_name: str = ""
    generated_at: str = ""
    seed: Optional[int] = None
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    model_version: Optional[str] = None
    git: GitProvenance = field(default_factory=GitProvenance)
    hardware: HardwareProvenance = field(default_factory=HardwareProvenance)
    dataset: DatasetProvenance = field(default_factory=DatasetProvenance)
    library_versions: Dict[str, Optional[str]] = field(default_factory=dict)
    config_snapshots: Dict[str, str] = field(default_factory=dict)
    configs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns the record as a JSON-serializable dictionary."""
        return asdict(self)

    def write(self, output_path: str) -> str:
        """Writes the record to a JSON file.

        Args:
            output_path: Destination ``metadata.json`` path.

        Returns:
            str: The path written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return output_path


def load_provenance(path: str) -> RunProvenance:
    """Reads a provenance record previously written by :meth:`RunProvenance.write`.

    Args:
        path: Path of the ``metadata.json`` document.

    Returns:
        RunProvenance: The reconstructed record. Unknown keys are ignored so
        a record written by a newer schema version still loads.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    nested = {
        "git": GitProvenance,
        "hardware": HardwareProvenance,
        "dataset": DatasetProvenance,
    }
    kwargs: Dict[str, Any] = {}
    known = {f.name for f in fields(RunProvenance)}
    for key, value in payload.items():
        if key not in known:
            continue
        if key in nested and isinstance(value, dict):
            sub_known = {f.name for f in fields(nested[key])}
            kwargs[key] = nested[key](
                **{k: v for k, v in value.items() if k in sub_known}
            )
        else:
            kwargs[key] = value
    return RunProvenance(**kwargs)


def _run_git(args: List[str], repo_dir: str) -> Optional[str]:
    """Runs a git command, returning its stripped stdout or ``None``."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"git command unavailable: {exc}")
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def collect_git_provenance(repo_dir: str = ".") -> GitProvenance:
    """Collects the git state of the working tree.

    Args:
        repo_dir: Directory inside the repository to inspect.

    Returns:
        GitProvenance: The collected state, with unavailable fields ``None``.
    """
    commit = _run_git(["rev-parse", "HEAD"], repo_dir)
    if commit is None:
        return GitProvenance()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    status = _run_git(["status", "--porcelain"], repo_dir)
    return GitProvenance(commit=commit, branch=branch, dirty=bool(status))


def collect_hardware_provenance() -> HardwareProvenance:
    """Collects hardware, platform and CUDA information.

    Returns:
        HardwareProvenance: The collected description.
    """
    hardware = HardwareProvenance(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
    )

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        return hardware

    hardware.cuda_version = torch.version.cuda
    hardware.cuda_available = bool(torch.cuda.is_available())
    if hardware.cuda_available:
        cudnn_version = torch.backends.cudnn.version()
        hardware.cudnn_version = (
            str(cudnn_version) if cudnn_version is not None else None
        )
        hardware.gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return hardware


def collect_library_versions(
    distributions: Tuple[str, ...] = TRACKED_LIBRARIES,
) -> Dict[str, Optional[str]]:
    """Collects installed versions of the tracked distributions.

    Args:
        distributions: Distribution names to look up.

    Returns:
        Dict[str, Optional[str]]: Mapping of name to version, ``None`` when
        the distribution is not installed.
    """
    versions: Dict[str, Optional[str]] = {}
    for name in distributions:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """Reads a JSON document, returning ``None`` when unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(f"Could not read '{path}': {exc}")
        return None
    return data if isinstance(data, dict) else None


def collect_dataset_provenance(
    raw_data_path: Optional[str] = None,
    metadata_dir: Optional[str] = None,
    processed_data_dir: Optional[str] = None,
    target_column: Optional[str] = None,
) -> DatasetProvenance:
    """Collects the dataset version and shape from cached artifacts.

    The fingerprint is read from the dataset metadata cache and the graph
    manifest rather than recomputed, so collecting provenance never re-reads
    the multi-gigabyte raw dataset.

    Args:
        raw_data_path: Configured raw dataset path.
        metadata_dir: Directory holding ``dataset_metadata.json``.
        processed_data_dir: Directory holding the processed shards.
        target_column: Name of the regression target column.

    Returns:
        DatasetProvenance: The collected record.
    """
    provenance = DatasetProvenance(
        raw_data_path=raw_data_path,
        processed_data_dir=processed_data_dir,
        target_column=target_column,
    )

    if metadata_dir:
        metadata = _read_json(os.path.join(metadata_dir, "dataset_metadata.json"))
        if metadata:
            provenance.fingerprint = metadata.get("dataset_fingerprint")
            stats = metadata.get("stats")
            if isinstance(stats, dict):
                provenance.total_records = stats.get("total_records")
                provenance.valid_records = stats.get("valid_records")

    if processed_data_dir:
        manifest = _read_json(
            os.path.join(processed_data_dir, "graphs", "manifest.json")
        )
        if manifest:
            provenance.fingerprint = manifest.get("fingerprint", provenance.fingerprint)
            provenance.successful_graphs = manifest.get("successful_graphs")
            provenance.failed_graphs = manifest.get("failed_graphs")
            provenance.shards_created = manifest.get("shards_created")
            graph_config = manifest.get("graph_config")
            if isinstance(graph_config, dict):
                provenance.graph_config = graph_config
            if provenance.total_records is None:
                provenance.total_records = manifest.get("total_records_processed")

    return provenance


def file_sha256(path: str, chunk_size: int = 1048576) -> Optional[str]:
    """Computes the SHA-256 digest of a file.

    Args:
        path: File to digest.
        chunk_size: Read size in bytes.

    Returns:
        Optional[str]: The hex digest, or ``None`` when the file cannot be
        read.
    """
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        logger.debug(f"Could not digest '{path}': {exc}")
        return None
    return digest.hexdigest()


def collect_provenance(
    run_id: str,
    experiment_name: str = "",
    seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    configs: Optional[Dict[str, Any]] = None,
    config_snapshots: Optional[Dict[str, str]] = None,
    raw_data_path: Optional[str] = None,
    metadata_dir: Optional[str] = None,
    processed_data_dir: Optional[str] = None,
    target_column: Optional[str] = None,
    repo_dir: str = ".",
    now: Optional[datetime] = None,
) -> RunProvenance:
    """Collects the full provenance record for a run (PRD §17).

    Args:
        run_id: The run identifier.
        experiment_name: Configured experiment name.
        seed: Global random seed for the run.
        checkpoint_path: Checkpoint the run was evaluated from.
        configs: Configuration values the run used, as plain dictionaries.
        config_snapshots: Mapping of snapshot name to its path in the run.
        raw_data_path: Configured raw dataset path.
        metadata_dir: Directory holding the dataset metadata cache.
        processed_data_dir: Directory holding the processed shards.
        target_column: Name of the regression target column.
        repo_dir: Directory inside the repository to inspect for git state.
        now: Collection timestamp. Defaults to the current UTC time.

    Returns:
        RunProvenance: The collected record.
    """
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    return RunProvenance(
        run_id=run_id,
        experiment_name=experiment_name,
        generated_at=moment.isoformat(),
        seed=seed,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=(file_sha256(checkpoint_path) if checkpoint_path else None),
        model_version=(os.path.basename(checkpoint_path) if checkpoint_path else None),
        git=collect_git_provenance(repo_dir),
        hardware=collect_hardware_provenance(),
        dataset=collect_dataset_provenance(
            raw_data_path=raw_data_path,
            metadata_dir=metadata_dir,
            processed_data_dir=processed_data_dir,
            target_column=target_column,
        ),
        library_versions=collect_library_versions(),
        config_snapshots=dict(config_snapshots or {}),
        configs=dict(configs or {}),
    )
