"""Domain-specific exceptions for the deterministic SEAL-Med core."""


class SealMedError(Exception):
    """Base class for deterministic SEAL-Med failures."""


class SchemaValidationError(SealMedError, ValueError):
    """Raised when a typed object violates a syntactic invariant."""


class DuplicateIdentifierError(SealMedError, ValueError):
    """Raised when an identifier is reused for a different object."""


class UnknownFactError(SealMedError, KeyError):
    """Raised when a claim or assessment references an unknown case fact."""


class UnknownCandidateError(SealMedError, KeyError):
    """Raised when a claim or assessment references an unknown diagnosis."""


class UnknownClaimError(SealMedError, KeyError):
    """Raised when a provenance edge references an unknown claim."""


class ClaimCycleError(SealMedError, ValueError):
    """Raised when a claim provenance edge would create a cycle."""


class StaleSnapshotError(SealMedError, RuntimeError):
    """Raised when a derived artifact no longer matches ledger state."""
