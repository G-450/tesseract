"""Custom exception hierarchy for the explainability subsystem.

All explainability errors inherit from ExplainabilityError, which
integrates with the project's broader ProjectError hierarchy.
"""


class ExplainabilityError(Exception):
    """Base exception for all explainability-related failures.

    Raised when the explainability subsystem encounters an unrecoverable
    error that cannot be isolated to a single sample.
    """


class ExplainerInitializationError(ExplainabilityError):
    """Raised when an explainer fails to initialize.

    Args:
        method: The name of the explainability method that failed.
        reason: A human-readable description of why initialization failed.
    """

    def __init__(self, method: str, reason: str) -> None:
        super().__init__(f"Explainer '{method}' failed to initialize. Reason: {reason}")
        self.method = method
        self.reason = reason


class SampleExplanationError(ExplainabilityError):
    """Raised when explanation fails for a specific sample.

    This error is intentionally isolated — it must never terminate
    explanation generation for the rest of the dataset (PRD §88).

    Args:
        sample_id: Identifier for the sample that failed.
        reason: A human-readable description of why explanation failed.
    """

    def __init__(self, sample_id: object, reason: str) -> None:
        super().__init__(
            f"Explanation failed for sample '{sample_id}'. Reason: {reason}"
        )
        self.sample_id = sample_id
        self.reason = reason


class ExportError(ExplainabilityError):
    """Raised when exporting an explanation result fails.

    Args:
        path: The file path where export was attempted.
        reason: A human-readable description of why export failed.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to export explanation to '{path}'. Reason: {reason}")
        self.path = path
        self.reason = reason
