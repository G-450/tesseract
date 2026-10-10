"""Experiment configuration and run identity (PRD §17, §18, §122).

A run is identified by a ``run_id``. Every artifact produced for that run
lives under a single directory derived from it, so runs can never overwrite
one another. The ``run_id`` is either pinned in ``configs/experiment.yaml``
(or on the command line) for a reproducible rerun, or generated from the
experiment name and the current UTC time.
"""

import os
import re
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, TypeVar

import yaml  # type: ignore

from src.config.exceptions import (
    ConfigFileError,
    ConfigurationError,
    UnknownConfigKeyError,
)

RUN_ID_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
"""UTC timestamp format embedded in generated run IDs."""

_UNSAFE_RUN_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

T = TypeVar("T")


@dataclass
class LoggingConfig:
    """Logging settings applied by the CLI entry points.

    Attributes:
        level: Root log level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    """

    level: str = "INFO"


@dataclass
class ExperimentConfig:
    """Identity, seed and artifact layout of a single experiment run.

    Attributes:
        name: Human-readable experiment name, used as the run ID prefix.
        seed: Global random seed recorded in the run's provenance metadata.
        run_id: Fixed run identifier, or ``None`` to generate one per run.
        experiments_dir: Root directory holding every experiment artifact.
        runs_dirname: Subdirectory of ``experiments_dir`` holding per-run
            directories.
        baseline_mae_ev_atom: Published CGCNN baseline MAE used for the
            benchmark comparison (PRD §99).
        logging: Logging settings.
    """

    name: str = "cgcnn_mp_e_form"
    seed: int = 42
    run_id: Optional[str] = None
    experiments_dir: str = "experiments"
    runs_dirname: str = "runs"
    baseline_mae_ev_atom: float = 0.039
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def runs_root(self) -> str:
        """Directory containing every structured run directory."""
        return os.path.join(self.experiments_dir, self.runs_dirname)

    def resolve_run_id(self, override: Optional[str] = None) -> str:
        """Returns the run ID for this run, generating one when unpinned.

        Args:
            override: Run ID supplied on the command line. Takes precedence
                over the configured ``run_id``.

        Returns:
            str: A filesystem-safe run identifier.
        """
        candidate = override or self.run_id
        if candidate:
            return sanitize_run_id(candidate)
        return generate_run_id(self.name)


def sanitize_run_id(run_id: str) -> str:
    """Reduces a run ID to a safe single path segment.

    Path separators and other characters that are awkward in directory names
    are collapsed to underscores, so a run ID can never escape the runs root.

    Args:
        run_id: The proposed run identifier.

    Returns:
        str: The sanitized identifier.

    Raises:
        ConfigurationError: If nothing usable remains after sanitization.
    """
    cleaned = _UNSAFE_RUN_ID_CHARS.sub("_", run_id.strip()).strip("._-")
    if not cleaned:
        raise ConfigurationError(f"Run ID '{run_id}' contains no usable characters.")
    return cleaned


def generate_run_id(name: str, now: Optional[datetime] = None) -> str:
    """Builds a run ID from an experiment name and a UTC timestamp.

    Args:
        name: Experiment name used as the identifier prefix.
        now: Timestamp to embed. Defaults to the current UTC time.

    Returns:
        str: An identifier of the form ``"<name>_<YYYYMMDDTHHMMSSZ>"``.
    """
    moment = now or datetime.now(timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime(RUN_ID_TIMESTAMP_FORMAT)
    return sanitize_run_id(f"{name}_{stamp}")


def _build(cls: Type[T], data: Dict[str, Any], section: str) -> T:
    """Instantiates a config dataclass, rejecting unknown keys (PRD §122).

    Args:
        cls: The dataclass to instantiate.
        data: Mapping of field names to values.
        section: Dotted section name used in error messages.

    Returns:
        T: The populated dataclass instance.

    Raises:
        UnknownConfigKeyError: If ``data`` holds keys outside the schema.
    """
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - allowed
    if unknown:
        raise UnknownConfigKeyError(section, list(unknown), list(allowed))
    return cls(**data)


def load_experiment_config(
    config_path: str = "configs/experiment.yaml",
) -> ExperimentConfig:
    """Loads the experiment configuration from a YAML file.

    A missing file yields the documented defaults; a malformed one is an
    error, because a run configured from a file that could not be parsed is
    not reproducible.

    Args:
        config_path: Path to the experiment YAML configuration.

    Returns:
        ExperimentConfig: The loaded configuration.

    Raises:
        ConfigFileError: If the file cannot be read or parsed.
        UnknownConfigKeyError: If the file contains unrecognised keys.
    """
    if not os.path.exists(config_path):
        return ExperimentConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigFileError(config_path, str(exc)) from exc

    if not data:
        return ExperimentConfig()
    if not isinstance(data, dict):
        raise ConfigFileError(config_path, "top-level document is not a mapping")

    unknown_top = set(data) - {"experiment"}
    if unknown_top:
        raise UnknownConfigKeyError("<root>", list(unknown_top), ["experiment"])

    experiment_data = dict(data.get("experiment") or {})
    logging_data = dict(experiment_data.pop("logging", None) or {})
    logging_config = _build(LoggingConfig, logging_data, "experiment.logging")

    return _build(
        ExperimentConfig,
        {**experiment_data, "logging": logging_config},
        "experiment",
    )
