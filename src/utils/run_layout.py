"""Canonical on-disk layout for a single experiment run (PRD §18, §19, §98).

Every artifact belonging to a run lives under one directory named after the
run ID, so a run can be inspected, archived or reproduced without ambiguity
and can never overwrite another run.

Layout::

    experiments/runs/<run_id>/
        config/            dataset.yaml, model.yaml, experiment.yaml snapshots
        checkpoints/       best.ckpt, last.ckpt
        logs/              training.log and CSV / TensorBoard logger output
        plots/             publication-quality figures
        explanations/      per-material explainability artifacts
        predictions/       inference predictions for external structures
        metrics.json       test-set metrics and benchmark comparison
        predictions.csv    per-sample test-set predictions
        summary.md         evaluation summary
        metadata.json      run provenance
        report.md          automated experiment report

The legacy top-level ``experiments/`` directories predate this layout and are
documented in ``experiments/README.md``.
"""

import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

SUBDIRECTORIES: tuple[str, ...] = (
    "config",
    "checkpoints",
    "logs",
    "plots",
    "explanations",
    "predictions",
)
"""Directories created inside every run directory."""


class RunDirectoryExistsError(OSError):
    """Raised when a run directory already exists and may not be reused.

    Args:
        run_dir: The conflicting directory.
    """

    def __init__(self, run_dir: str) -> None:
        super().__init__(
            f"Run directory '{run_dir}' already exists. Choose a different "
            f"--run-id or pass --overwrite to replace it."
        )
        self.run_dir = run_dir


@dataclass(frozen=True)
class RunLayout:
    """Resolved paths of every artifact slot for one run.

    Attributes:
        run_id: The run identifier naming the directory.
        root: Absolute path of the run directory.
    """

    run_id: str
    root: str

    @classmethod
    def for_run(cls, runs_root: str, run_id: str) -> "RunLayout":
        """Returns the layout for a run without touching the filesystem.

        Args:
            runs_root: Directory holding every run directory.
            run_id: The run identifier.

        Returns:
            RunLayout: The resolved layout.
        """
        return cls(run_id=run_id, root=os.path.abspath(os.path.join(runs_root, run_id)))

    # -- Directories ------------------------------------------------------

    @property
    def config_dir(self) -> str:
        """Directory holding the configuration snapshots for the run."""
        return os.path.join(self.root, "config")

    @property
    def checkpoints_dir(self) -> str:
        """Directory holding model checkpoints."""
        return os.path.join(self.root, "checkpoints")

    @property
    def logs_dir(self) -> str:
        """Directory holding training logs and logger output."""
        return os.path.join(self.root, "logs")

    @property
    def plots_dir(self) -> str:
        """Directory holding generated figures."""
        return os.path.join(self.root, "plots")

    @property
    def explanations_dir(self) -> str:
        """Directory holding explainability artifacts."""
        return os.path.join(self.root, "explanations")

    @property
    def predictions_dir(self) -> str:
        """Directory holding inference predictions."""
        return os.path.join(self.root, "predictions")

    # -- Files ------------------------------------------------------------

    @property
    def metrics_json(self) -> str:
        """Path of the test-set metrics file."""
        return os.path.join(self.root, "metrics.json")

    @property
    def predictions_csv(self) -> str:
        """Path of the per-sample test-set predictions file."""
        return os.path.join(self.root, "predictions.csv")

    @property
    def summary_md(self) -> str:
        """Path of the evaluation summary."""
        return os.path.join(self.root, "summary.md")

    @property
    def metadata_json(self) -> str:
        """Path of the run provenance metadata."""
        return os.path.join(self.root, "metadata.json")

    @property
    def report_md(self) -> str:
        """Path of the automated experiment report."""
        return os.path.join(self.root, "report.md")

    @property
    def training_log(self) -> str:
        """Path of the plain-text training log."""
        return os.path.join(self.logs_dir, "training.log")

    def directories(self) -> List[str]:
        """Returns every directory belonging to this run, root first."""
        return [self.root] + [os.path.join(self.root, name) for name in SUBDIRECTORIES]

    def create(self, exist_ok: bool = True, overwrite: bool = False) -> "RunLayout":
        """Creates the run directory tree.

        Args:
            exist_ok: Allow writing into an existing run directory.
            overwrite: Delete an existing run directory first. Takes
                precedence over ``exist_ok``.

        Returns:
            RunLayout: This layout, for chaining.

        Raises:
            RunDirectoryExistsError: If the directory exists while
                ``exist_ok`` and ``overwrite`` are both false.
        """
        if os.path.exists(self.root):
            if overwrite:
                shutil.rmtree(self.root)
            elif not exist_ok:
                raise RunDirectoryExistsError(self.root)

        for directory in self.directories():
            os.makedirs(directory, exist_ok=True)
        return self

    def snapshot_configs(self, config_paths: Dict[str, str]) -> Dict[str, str]:
        """Copies configuration files into the run's ``config/`` directory.

        Snapshotting the configuration is what makes a report reproducible:
        the run records the configuration it actually used, independent of
        later edits to ``configs/`` (PRD §17).

        Args:
            config_paths: Mapping of snapshot name (without extension) to the
                source configuration file path.

        Returns:
            Dict[str, str]: Mapping of snapshot name to the written path, for
            the sources that existed.
        """
        os.makedirs(self.config_dir, exist_ok=True)
        written: Dict[str, str] = {}
        for name, source in config_paths.items():
            if not source or not os.path.isfile(source):
                continue
            extension = os.path.splitext(source)[1] or ".yaml"
            destination = os.path.join(self.config_dir, f"{name}{extension}")
            if os.path.abspath(source) != destination:
                shutil.copyfile(source, destination)
            written[name] = destination
        return written

    def relative(self, path: Optional[str]) -> Optional[str]:
        """Expresses a path relative to the run root, for use in reports.

        Args:
            path: An absolute or relative filesystem path, or ``None``.

        Returns:
            Optional[str]: The POSIX-style path relative to the run root, or
            the unchanged input when it lies outside the run directory.
        """
        if not path:
            return None
        absolute = os.path.abspath(path)
        try:
            relative = os.path.relpath(absolute, self.root)
        except ValueError:
            return path
        if relative.startswith(os.pardir):
            return path
        return relative.replace(os.sep, "/")
