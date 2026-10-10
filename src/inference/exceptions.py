"""Custom exception hierarchy for the inference subsystem.

All inference errors inherit from InferenceError, which integrates with
the project's broader exception hierarchy (PRD §123).
"""


class InferenceError(Exception):
    """Base exception for all inference-related failures.

    Raised when the inference subsystem encounters an unrecoverable error
    that cannot be isolated to a single structure.
    """


class ModelLoadError(InferenceError):
    """Raised when a trained checkpoint cannot be loaded for inference.

    Args:
        checkpoint_path: Path to the checkpoint that failed to load.
        reason: A human-readable description of why loading failed.
    """

    def __init__(self, checkpoint_path: str, reason: str) -> None:
        super().__init__(
            f"Failed to load checkpoint '{checkpoint_path}'. Reason: {reason}"
        )
        self.checkpoint_path = checkpoint_path
        self.reason = reason


class StructureLoadError(InferenceError):
    """Raised when a crystal structure cannot be parsed from disk.

    This error is intentionally isolated — during batch prediction it must
    never terminate processing of the remaining structures (PRD §91).

    Args:
        path: Path to the structure file that failed to parse.
        reason: A human-readable description of why parsing failed.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to load structure from '{path}'. Reason: {reason}")
        self.path = path
        self.reason = reason


class PredictionError(InferenceError):
    """Raised when prediction fails for a specific structure.

    This error is intentionally isolated — during batch prediction it must
    never terminate processing of the remaining structures (PRD §91).

    Args:
        sample_id: Identifier for the structure that failed.
        reason: A human-readable description of why prediction failed.
    """

    def __init__(self, sample_id: object, reason: str) -> None:
        super().__init__(
            f"Prediction failed for structure '{sample_id}'. Reason: {reason}"
        )
        self.sample_id = sample_id
        self.reason = reason


class PredictionExportError(InferenceError):
    """Raised when exporting predictions fails.

    Args:
        path: The file path where export was attempted.
        reason: A human-readable description of why export failed.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to export predictions to '{path}'. Reason: {reason}")
        self.path = path
        self.reason = reason
