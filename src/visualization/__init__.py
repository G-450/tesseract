"""Visualization package for CGCNN evaluation and results analysis."""

from src.visualization.plots import (
    DEFAULT_DPI,
    PUBLICATION_RC_PARAMS,
    apply_publication_style,
    generate_all_plots,
    plot_learning_rate,
    plot_loss_curves,
    plot_parity,
    plot_prediction_histogram,
    plot_residuals,
)

__all__ = [
    "plot_parity",
    "plot_residuals",
    "plot_loss_curves",
    "plot_learning_rate",
    "plot_prediction_histogram",
    "generate_all_plots",
    "apply_publication_style",
    "PUBLICATION_RC_PARAMS",
    "DEFAULT_DPI",
]
