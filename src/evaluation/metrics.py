"""Evaluation metrics for formation energy prediction."""

from typing import Dict

import torch


def _validate_inputs(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validates and flattens input prediction and target tensors.

    Args:
        predictions: Predicted tensor of shape (N,) or (N, 1).
        targets: Ground truth tensor of shape (N,) or (N, 1).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Flattened 1D float32 tensors.

    Raises:
        ValueError: If tensors have differing number of elements or are empty.
        TypeError: If inputs are not torch.Tensor instances.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(
        targets, torch.Tensor
    ):
        raise TypeError(
            f"Expected torch.Tensor inputs, got {type(predictions).__name__} "
            f"and {type(targets).__name__}."
        )

    pred_flat = predictions.detach().view(-1).float()
    target_flat = targets.detach().view(-1).float()

    if pred_flat.numel() == 0 or target_flat.numel() == 0:
        raise ValueError("Cannot compute metrics on empty tensors.")

    if pred_flat.numel() != target_flat.numel():
        raise ValueError(
            f"Number of predictions ({pred_flat.numel()}) does not match "
            f"number of targets ({target_flat.numel()})."
        )

    return pred_flat, target_flat


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Mean Absolute Error (MAE) between predictions and targets.

    Args:
        predictions: Model predictions tensor.
        targets: Ground truth target values tensor.

    Returns:
        float: Mean absolute error in eV/atom.

    Raises:
        ValueError: If input tensors are empty or shape-mismatched.
        TypeError: If inputs are not torch tensors.
    """
    preds, trues = _validate_inputs(predictions, targets)
    mae = torch.mean(torch.abs(preds - trues)).item()
    return float(mae)


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Root Mean Square Error (RMSE) between predictions and targets.

    Args:
        predictions: Model predictions tensor.
        targets: Ground truth target values tensor.

    Returns:
        float: Root mean square error in eV/atom.

    Raises:
        ValueError: If input tensors are empty or shape-mismatched.
        TypeError: If inputs are not torch tensors.
    """
    preds, trues = _validate_inputs(predictions, targets)
    mse = torch.mean((preds - trues) ** 2)
    rmse = torch.sqrt(mse).item()
    return float(rmse)


def compute_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Coefficient of Determination (R²) score.

    Args:
        predictions: Model predictions tensor.
        targets: Ground truth target values tensor.

    Returns:
        float: R² score (1.0 is perfect prediction).

    Raises:
        ValueError: If input tensors are empty or shape-mismatched.
        TypeError: If inputs are not torch tensors.
    """
    preds, trues = _validate_inputs(predictions, targets)
    ss_res = torch.sum((trues - preds) ** 2)
    ss_tot = torch.sum((trues - torch.mean(trues)) ** 2)

    if ss_tot.item() == 0.0:
        # Constant target edge-case
        return 1.0 if ss_res.item() == 0.0 else 0.0

    r2 = 1.0 - (ss_res / ss_tot).item()
    return float(r2)


def compute_bias(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Mean Prediction Bias (mean signed error: mean(predictions - targets)).

    A positive bias indicates systematic overestimation, while a negative
    bias indicates systematic underestimation.

    Args:
        predictions: Model predictions tensor.
        targets: Ground truth target values tensor.

    Returns:
        float: Mean prediction bias in eV/atom.

    Raises:
        ValueError: If input tensors are empty or shape-mismatched.
        TypeError: If inputs are not torch tensors.
    """
    preds, trues = _validate_inputs(predictions, targets)
    bias = torch.mean(preds - trues).item()
    return float(bias)


def compute_all_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """Computes all core evaluation metrics.

    Args:
        predictions: Model predictions tensor.
        targets: Ground truth target values tensor.

    Returns:
        Dict[str, float]: Dictionary with 'mae', 'rmse', 'r2', and 'bias'.
    """
    return {
        "mae": compute_mae(predictions, targets),
        "rmse": compute_rmse(predictions, targets),
        "r2": compute_r2(predictions, targets),
        "bias": compute_bias(predictions, targets),
    }
