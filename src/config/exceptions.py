"""Exception hierarchy for the configuration subsystem (PRD §123).

Configuration failures are raised rather than silently tolerated: an
experiment whose configuration could not be read exactly as written is not
reproducible, so the pipeline must stop instead of falling back to defaults.
"""


class ConfigurationError(Exception):
    """Base exception for every configuration failure.

    Integrates with the project's broader ``ProjectError`` hierarchy
    (PRD §123).
    """


class ConfigFileError(ConfigurationError):
    """Raised when a configuration file cannot be read or parsed.

    Args:
        config_path: Path to the configuration file that failed to load.
        reason: Human-readable description of why loading failed.
    """

    def __init__(self, config_path: str, reason: str) -> None:
        super().__init__(f"Failed to load config '{config_path}'. Reason: {reason}")
        self.config_path = config_path
        self.reason = reason


class UnknownConfigKeyError(ConfigurationError):
    """Raised when a configuration file contains unrecognised keys.

    Unknown keys are almost always typos or stale options; silently ignoring
    them would make a run differ from its configuration (PRD §122).

    Args:
        section: Dotted name of the configuration section, e.g.
            ``"experiment.logging"``.
        unknown_keys: The keys that are not part of the schema.
        allowed_keys: The keys the section accepts.
    """

    def __init__(
        self,
        section: str,
        unknown_keys: list[str],
        allowed_keys: list[str],
    ) -> None:
        super().__init__(
            f"Unknown key(s) {sorted(unknown_keys)} in config section "
            f"'{section}'. Allowed keys: {sorted(allowed_keys)}."
        )
        self.section = section
        self.unknown_keys = sorted(unknown_keys)
        self.allowed_keys = sorted(allowed_keys)
