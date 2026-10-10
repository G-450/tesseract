"""Checkpoint deserialization helpers (PRD 10.10).

Lightning checkpoints store their hyperparameters as pickled project
dataclasses. From PyTorch 2.6 onwards ``torch.load`` defaults to
``weights_only=True``, which rejects any non-tensor global, so loading a
checkpoint fails unless those classes are allowlisted.

This module provides the single allowlist shared by every subsystem that
loads a checkpoint (evaluation, explainability and inference), so the set of
permitted globals is defined in exactly one place.
"""

import contextlib
from typing import Any, ContextManager, Tuple

import torch

from src.config.model_config import CGCNNConfig, ModelConfig, OptimizerConfig

# Only the project's own configuration classes are permitted; every other
# global stays blocked.
CHECKPOINT_SAFE_GLOBALS: Tuple[type, ...] = (
    CGCNNConfig,
    ModelConfig,
    OptimizerConfig,
)


def checkpoint_safe_globals() -> ContextManager[Any]:
    """Returns a context manager allowlisting the project's config classes.

    Falls back to a no-op on PyTorch versions predating
    ``torch.serialization.safe_globals``, where ``torch.load`` does not
    restrict globals by default.

    Returns:
        ContextManager[Any]: Scope within which a checkpoint's pickled
        configuration dataclasses can be deserialized.
    """
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        return contextlib.nullcontext()
    allowlist: ContextManager[Any] = safe_globals(list(CHECKPOINT_SAFE_GLOBALS))
    return allowlist
