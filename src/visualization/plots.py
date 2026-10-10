"""Visualization module for model evaluation and publication-quality plots."""

import glob
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must be called before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.evaluator import EvaluationResult  # noqa: E402

DEFAULT_DPI = 300
"""Export resolution for every figure (PRD §97)."""

PUBLICATION_RC_PARAMS: Dict[str, Any] = {
    "figure.dpi": 100,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "axes.edgecolor": "#334155",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CBD5E1",
    "lines.linewidth": 2.0,
    "figure.autolayout": False,
}
"""Explicit figure styling, so no figure relies on matplotlib defaults.

Every entry is set deliberately: PRD §97 requires consistent font sizing and
high-DPI export across experiments, which default styling does not guarantee
between matplotlib versions.
"""

# Series are distinguished by both colour and dash pattern so every figure
# stays readable when printed in greyscale (PRD §94).
COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#EF4444"
COLOR_TERTIARY = "#047857"
COLOR_REFERENCE = "#111827"
COLOR_BOX_FACE = "#F8FAFC"
COLOR_BOX_EDGE = "#CBD5E1"

_ANNOTATION_BOX = dict(
    boxstyle="round,pad=0.5",
    facecolor=COLOR_BOX_FACE,
    edgecolor=COLOR_BOX_EDGE,
    alpha=0.9,
)

_LEARNING_RATE_COLUMNS = ("lr-Adam", "lr", "learning_rate", "lr-Adam/pg1")


def apply_publication_style() -> None:
    """Applies the project's explicit figure styling to matplotlib.

    Called once at import so every figure in the project shares one visual
    language regardless of the caller (PRD §94, §97).
    """
    plt.rcParams.update(PUBLICATION_RC_PARAMS)  # type: ignore[arg-type]


apply_publication_style()


def _to_numpy(data: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """Converts torch tensor, list, or array to a 1D numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().view(-1).numpy().astype(np.float64)
    if isinstance(data, list):
        return np.asarray(data, dtype=np.float64).ravel()
    return np.asarray(data, dtype=np.float64).ravel()


def plot_parity(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    output_path: str,
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Parity Plot: Predicted vs. True Formation Energy",
    dpi: int = 300,
) -> str:
    """Generates and saves a publication-quality parity plot (predicted vs. true).

    Args:
        predictions: Model predictions.
        targets: Ground truth target values.
        output_path: Path where the PNG image will be saved.
        metrics: Optional dictionary containing 'mae', 'rmse', and 'r2'.
        title: Plot title.
        dpi: Dots per inch resolution.

    Returns:
        str: Absolute path to the saved plot image.
    """
    preds = _to_numpy(predictions)
    trues = _to_numpy(targets)

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=dpi)

    # Determine plot bounds with padding
    min_val = min(float(np.min(trues)), float(np.min(preds)))
    max_val = max(float(np.max(trues)), float(np.max(preds)))
    span = max_val - min_val
    padding = span * 0.05 if span > 0 else 0.5
    lims = [min_val - padding, max_val + padding]

    # Diagonal reference line (y = x)
    ax.plot(
        lims,
        lims,
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label="Perfect Parity ($y = x$)",
        zorder=2,
    )

    # Scatter points with slight transparency and edge
    ax.scatter(
        trues,
        preds,
        color="#2563EB",
        alpha=0.6,
        edgecolors="none",
        s=30,
        label="Test Samples",
        zorder=3,
    )

    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[0], lims[1])
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("True Formation Energy (eV/atom)", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        "Predicted Formation Energy (eV/atom)", fontsize=11, fontweight="bold"
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Metrics annotation box
    if metrics is not None:
        mae = metrics.get("mae", float(np.mean(np.abs(preds - trues))))
        rmse = metrics.get("rmse", float(np.sqrt(np.mean((preds - trues) ** 2))))
        r2 = metrics.get("r2")
        textstr = (
            f"$N = {len(preds):,}$\nMAE = {mae:.4f} eV/atom\nRMSE = {rmse:.4f} eV/atom"
        )
        if r2 is not None:
            textstr += f"\n$R^2 = {r2:.4f}$"

        props = dict(
            boxstyle="round,pad=0.5",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            alpha=0.9,
        )
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            zorder=4,
        )

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def plot_residuals(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    output_path: str,
    title: str = "Residual Distribution (Predicted - True)",
    dpi: int = 300,
) -> str:
    """Generates and saves a residual distribution histogram with summary statistics.

    Args:
        predictions: Model predictions.
        targets: Ground truth target values.
        output_path: Path where the PNG image will be saved.
        title: Plot title.
        dpi: Dots per inch resolution.

    Returns:
        str: Absolute path to the saved plot image.
    """
    preds = _to_numpy(predictions)
    trues = _to_numpy(targets)
    residuals = preds - trues

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    mean_bias = float(np.mean(residuals))
    std_dev = float(np.std(residuals))

    # Histogram
    counts, bins, patches = ax.hist(
        residuals,
        bins=min(50, max(10, len(residuals) // 5)),
        density=True,
        color="#3B82F6",
        edgecolor="#1D4ED8",
        alpha=0.7,
        label="Residual Density",
    )

    # Fit Gaussian curve overlay if variance > 0
    if std_dev > 1e-7:
        x_grid = np.linspace(min(residuals), max(residuals), 200)
        gaussian_fit = (1.0 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_grid - mean_bias) / std_dev) ** 2
        )
        ax.plot(
            x_grid,
            gaussian_fit,
            color="#EF4444",
            linewidth=2,
            label=r"Gaussian Fit ($\mu=$"
            + f"{mean_bias:.3f}"
            + r", $\sigma=$"
            + f"{std_dev:.3f}$)",
        )

    # Zero residual reference line
    ax.axvline(
        0.0,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label="Zero Error",
    )

    ax.set_xlabel(r"Residual: $\hat{y} - y$ (eV/atom)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Summary box
    textstr = (
        f"Mean Bias = {mean_bias:+.4f} eV/atom\n"
        f"Std Dev = {std_dev:.4f} eV/atom\n"
        f"Median AE = {np.median(np.abs(residuals)):.4f} eV/atom"
    )
    props = dict(
        boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def _resolve_csv_log(log_path_or_dir: str) -> Optional[str]:
    """Resolves a CSV log path from a file or a logger directory.

    Args:
        log_path_or_dir: A ``metrics.csv`` path, or a directory containing one
            at any depth (PyTorch Lightning's ``CSVLogger`` nests it under a
            version directory).

    Returns:
        Optional[str]: The most recently modified matching CSV log, or
        ``None`` when none exists.
    """
    if os.path.isfile(log_path_or_dir):
        return log_path_or_dir
    if not os.path.isdir(log_path_or_dir):
        return None

    candidates = sorted(
        glob.glob(os.path.join(log_path_or_dir, "**", "metrics.csv"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_epoch_log(log_path_or_dir: str) -> Optional[pd.DataFrame]:
    """Loads a CSV training log aggregated by epoch.

    Aggregation is mean-per-epoch presentation formatting only: Lightning
    writes one row per logging step, which cannot be drawn directly against
    an epoch axis.

    Args:
        log_path_or_dir: A ``metrics.csv`` path or a directory containing one.

    Returns:
        Optional[pd.DataFrame]: Numeric columns indexed by epoch, or ``None``
        when no usable log was found.
    """
    csv_file = _resolve_csv_log(log_path_or_dir)
    if csv_file is None:
        return None

    try:
        df = pd.read_csv(csv_file)
    except (OSError, ValueError, pd.errors.ParserError):
        return None

    if "epoch" not in df.columns or df.empty:
        return None
    return df.groupby("epoch").mean(numeric_only=True)


def plot_prediction_histogram(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    output_path: str,
    title: str = "Formation Energy Distribution: Predicted vs. True",
    dpi: int = DEFAULT_DPI,
) -> str:
    """Generates and saves a predicted-vs-true value histogram (PRD §95).

    Overlaying the two distributions shows whether the model reproduces the
    spread of the target variable, not only its central tendency.

    Args:
        predictions: Model predictions.
        targets: Ground truth target values.
        output_path: Path where the PNG image will be saved.
        title: Plot title.
        dpi: Dots per inch resolution.

    Returns:
        str: Absolute path to the saved plot image.
    """
    preds = _to_numpy(predictions)
    trues = _to_numpy(targets)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    combined = np.concatenate([trues, preds]) if len(preds) else np.asarray([0.0])
    bin_count = int(min(60, max(10, len(preds) // 10))) if len(preds) else 10
    bins: List[float] = np.linspace(
        float(np.min(combined)), float(np.max(combined)), bin_count + 1
    ).tolist()

    ax.hist(
        trues,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=COLOR_REFERENCE,
        alpha=0.25,
        label="True",
    )
    ax.hist(
        trues,
        bins=bins,
        density=True,
        histtype="step",
        color=COLOR_REFERENCE,
        linewidth=1.8,
        linestyle="-",
    )
    ax.hist(
        preds,
        bins=bins,
        density=True,
        histtype="step",
        color=COLOR_PRIMARY,
        linewidth=2.0,
        linestyle="--",
        label="Predicted",
    )

    ax.set_xlabel("Formation Energy (eV/atom)")
    ax.set_ylabel("Probability Density")
    ax.set_title(title, pad=12)

    textstr = (
        f"$N = {len(preds):,}$\n"
        f"True: $\\mu=${np.mean(trues):+.3f}, $\\sigma=${np.std(trues):.3f}\n"
        f"Pred: $\\mu=${np.mean(preds):+.3f}, $\\sigma=${np.std(preds):.3f}"
    )
    ax.text(
        0.02,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=_ANNOTATION_BOX,
    )

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def plot_learning_rate(
    log_path_or_dir: str,
    output_path: str,
    title: str = "Learning Rate Schedule",
    dpi: int = DEFAULT_DPI,
) -> Optional[str]:
    """Generates the learning rate schedule from CSV training logs (PRD §95).

    Requires the training run to have logged the learning rate, which
    ``train_cli.py`` does through Lightning's ``LearningRateMonitor``. Logs
    from runs that predate that callback carry no such column.

    Args:
        log_path_or_dir: CSV log file path or directory containing metrics.csv.
        output_path: Path where the PNG image will be saved.
        title: Plot title.
        dpi: Dots per inch resolution.

    Returns:
        Optional[str]: Absolute path to the saved plot, or ``None`` when the
        log records no learning rate.
    """
    grouped = _load_epoch_log(log_path_or_dir)
    if grouped is None:
        return None

    column = next(
        (
            name
            for name in _LEARNING_RATE_COLUMNS
            if name in grouped.columns and not grouped[name].isna().all()
        ),
        None,
    )
    if column is None:
        column = next(
            (
                name
                for name in grouped.columns
                if str(name).startswith("lr") and not grouped[name].isna().all()
            ),
            None,
        )
    if column is None:
        return None

    series = grouped[column].dropna()
    if series.empty:
        return None
    epochs = np.asarray(series.index.to_numpy(), dtype=np.float64)
    rates = np.asarray(series.to_numpy(), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    ax.plot(
        epochs,
        rates,
        color=COLOR_TERTIARY,
        linestyle="-",
        marker="o",
        markersize=3,
        label=f"Learning Rate ({column})",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (dimensionless)")
    ax.set_title(title, pad=12)
    if float(np.min(rates)) > 0.0:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def plot_loss_curves(
    log_path_or_dir: str,
    output_path: str,
    title: str = "Training and Validation Loss Curves",
    dpi: int = 300,
) -> Optional[str]:
    """Generates training and validation loss curves from CSV logs.

    Args:
        log_path_or_dir: CSV log file path or directory containing metrics.csv.
        output_path: Path where the PNG image will be saved.
        title: Plot title.
        dpi: Dots per inch resolution.

    Returns:
        Optional[str]: Absolute path to the saved plot, or None if no log
        file was found.
    """
    grouped = _load_epoch_log(log_path_or_dir)
    if grouped is None:
        return None

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)

    has_train = (
        "train_loss" in grouped.columns and not grouped["train_loss"].isna().all()
    )
    has_val = "val_loss" in grouped.columns and not grouped["val_loss"].isna().all()

    if not (has_train or has_val):
        plt.close(fig)
        return None

    if has_train:
        train_data = grouped["train_loss"].dropna()
        ax1.plot(
            np.asarray(train_data.index),
            np.asarray(train_data.values),
            label="Train Loss (MSE)",
            color=COLOR_PRIMARY,
            linestyle="-",
            linewidth=2,
        )

    if has_val:
        val_data = grouped["val_loss"].dropna()
        ax1.plot(
            np.asarray(val_data.index),
            np.asarray(val_data.values),
            label="Val Loss (MSE)",
            color=COLOR_SECONDARY,
            linestyle="--",
            linewidth=2,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE, eV²/atom²)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary axis for val_MAE / val_mae if present
    mae_col = None
    if "val_MAE" in grouped.columns and not grouped["val_MAE"].isna().all():
        mae_col = "val_MAE"
    elif "val_mae" in grouped.columns and not grouped["val_mae"].isna().all():
        mae_col = "val_mae"

    if mae_col is not None:
        mae_data = grouped[mae_col].dropna()
        ax2 = ax1.twinx()
        ax2.plot(
            np.asarray(mae_data.index),
            np.asarray(mae_data.values),
            label="Val MAE",
            color=COLOR_TERTIARY,
            linestyle="-.",
            linewidth=1.8,
        )
        ax2.set_ylabel("Validation MAE (eV/atom)", color=COLOR_TERTIARY)
        ax2.tick_params(axis="y", labelcolor=COLOR_TERTIARY)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title(title, pad=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def generate_all_plots(
    result: EvaluationResult,
    output_dir: str,
    log_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generates every figure a completed experiment requires (PRD §95).

    Figures that depend on a training log are skipped when no log is
    available, so evaluating a checkpoint without its logs still produces the
    prediction-based figures rather than failing.

    Args:
        result: EvaluationResult containing predictions, targets, and metrics.
        output_dir: Directory where plots will be saved.
        log_dir: Optional CSV log file or directory containing training logs,
            used for the loss-curve and learning-rate figures.

    Returns:
        Dict[str, str]: Mapping of plot identifiers to saved file paths,
        containing only the figures that were generated.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots: Dict[str, str] = {}

    saved_plots["parity_plot"] = plot_parity(
        predictions=result.predictions,
        targets=result.targets,
        output_path=os.path.join(output_dir, "parity_plot.png"),
        metrics=result.metrics,
    )

    saved_plots["residual_plot"] = plot_residuals(
        predictions=result.predictions,
        targets=result.targets,
        output_path=os.path.join(output_dir, "residual_plot.png"),
    )

    saved_plots["prediction_histogram"] = plot_prediction_histogram(
        predictions=result.predictions,
        targets=result.targets,
        output_path=os.path.join(output_dir, "prediction_histogram.png"),
    )

    if log_dir is not None:
        loss_curve_path = plot_loss_curves(
            log_path_or_dir=log_dir,
            output_path=os.path.join(output_dir, "loss_curve.png"),
        )
        if loss_curve_path is not None:
            saved_plots["loss_curve"] = loss_curve_path

        learning_rate_path = plot_learning_rate(
            log_path_or_dir=log_dir,
            output_path=os.path.join(output_dir, "learning_rate.png"),
        )
        if learning_rate_path is not None:
            saved_plots["learning_rate"] = learning_rate_path

    return saved_plots
