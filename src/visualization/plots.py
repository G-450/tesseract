"""Visualization module for model evaluation and publication-quality plots."""

import glob
import os
from typing import Dict, Optional, Union

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must be called before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.evaluator import EvaluationResult  # noqa: E402


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
    csv_file = None
    if os.path.isfile(log_path_or_dir):
        csv_file = log_path_or_dir
    elif os.path.isdir(log_path_or_dir):
        # Look for metrics.csv in directory or subdirectories
        candidates = sorted(
            glob.glob(
                os.path.join(log_path_or_dir, "**", "metrics.csv"), recursive=True
            ),
            key=os.path.getmtime,
            reverse=True,
        )
        if candidates:
            csv_file = candidates[0]

    if csv_file is None or not os.path.exists(csv_file):
        return None

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return None

    if "epoch" not in df.columns:
        return None

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)

    # Aggregate by epoch
    grouped = df.groupby("epoch").mean(numeric_only=True)

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
            color="#3B82F6",
            linewidth=2,
        )

    if has_val:
        val_data = grouped["val_loss"].dropna()
        ax1.plot(
            np.asarray(val_data.index),
            np.asarray(val_data.values),
            label="Val Loss (MSE)",
            color="#EF4444",
            linewidth=2,
        )

    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss (MSE)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary axis for val_mae if present
    if "val_mae" in grouped.columns and not grouped["val_mae"].isna().all():
        mae_data = grouped["val_mae"].dropna()
        ax2 = ax1.twinx()
        ax2.plot(
            np.asarray(mae_data.index),
            np.asarray(mae_data.values),
            label="Val MAE",
            color="#10B981",
            linestyle="--",
            linewidth=1.8,
        )
        ax2.set_ylabel(
            "Validation MAE (eV/atom)", color="#10B981", fontsize=11, fontweight="bold"
        )
        ax2.tick_params(axis="y", labelcolor="#10B981")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9
        )
    else:
        ax1.legend(loc="upper right", framealpha=0.9)

    ax1.set_title(title, fontsize=12, fontweight="bold", pad=12)
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
    """Generates and exports all evaluation plots.

    Args:
        result: EvaluationResult containing predictions, targets, and metrics.
        output_dir: Directory where plots will be saved.
        log_dir: Optional directory containing training CSV logs for loss curves.

    Returns:
        Dict[str, str]: Mapping of plot identifiers to saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots: Dict[str, str] = {}

    parity_path = os.path.join(output_dir, "parity_plot.png")
    plot_parity(
        predictions=result.predictions,
        targets=result.targets,
        output_path=parity_path,
        metrics=result.metrics,
    )
    saved_plots["parity_plot"] = parity_path

    residual_path = os.path.join(output_dir, "residual_plot.png")
    plot_residuals(
        predictions=result.predictions,
        targets=result.targets,
        output_path=residual_path,
    )
    saved_plots["residual_plot"] = residual_path

    if log_dir is not None:
        loss_curve_path = os.path.join(output_dir, "loss_curve.png")
        saved_loss_path = plot_loss_curves(
            log_path_or_dir=log_dir,
            output_path=loss_curve_path,
        )
        if saved_loss_path is not None:
            saved_plots["loss_curve"] = saved_loss_path

    return saved_plots
