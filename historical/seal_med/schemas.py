"""Typed, model-independent data contracts for the SEAL-Med M1 core."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, TypeVar

from .exceptions import SchemaValidationError


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")


def _require_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or not _IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise SchemaValidationError(
            f"{name} must start with a letter and contain only letters, "
            f"numbers, '_', '.', ':' or '-': {value!r}"
        )


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must not be empty")


_EnumT = TypeVar("_EnumT", bound=Enum)


def _require_enum(name: str, value: object, enum_type: type[_EnumT]) -> None:
    if not isinstance(value, enum_type):
        raise SchemaValidationError(
            f"{name} must be a {enum_type.__name__}, got {type(value).__name__}"
        )


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise SchemaValidationError(f"{name} must be an integer")


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be numeric")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise SchemaValidationError(f"{name} must be finite")


def _as_unique_tuple(
    values: Iterable[str],
    *,
    name: str = "identifier collection",
    identifiers: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise SchemaValidationError(
            f"{name} must be an array, not a single string"
        )
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise SchemaValidationError(
                f"{name} may contain strings only"
            )
        stripped = value.strip()
        if not stripped:
            raise SchemaValidationError(f"{name} may not contain blank values")
        if identifiers:
            _require_identifier(name, stripped)
        normalized.append(stripped)
    return tuple(dict.fromkeys(normalized))


class StrEnum(str, Enum):
    """A Python 3.11-compatible string enum with readable serialization."""

    def __str__(self) -> str:
        return self.value


class FactType(StrEnum):
    HISTORY = "history"
    SYMPTOM = "symptom"
    PHYSICAL_EXAM = "physical_exam"
    VITAL = "vital"
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    PATHOLOGY = "pathology"
    MEDICATION = "medication"
    DEMOGRAPHIC = "demographic"
    NEGATIVE_FINDING = "negative_finding"
    OTHER = "other"


class Polarity(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class Temporality(StrEnum):
    CURRENT = "current"
    PAST = "past"
    FUTURE = "future"
    UNKNOWN = "unknown"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    OTHER = "other"


class CandidateSource(StrEnum):
    MCQ_OPTION = "mcq_option"
    AGENT_PROPOSED = "agent_proposed"
    RISK_PROPOSED = "risk_proposed"


class ClaimType(StrEnum):
    SOURCE_OBSERVATION = "source_observation"
    DIAGNOSTIC_HYPOTHESIS = "diagnostic_hypothesis"
    MEDICAL_RULE = "medical_rule"
    COMPLICATION = "complication"
    RISK_ITEM = "risk_item"
    RECOMMENDED_ACTION = "recommended_action"


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    SOURCE_BOUND = "source_bound"
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    EXPIRED = "expired"


class TaintState(StrEnum):
    CLEAN = "clean"
    TAINTED = "tainted"
    QUARANTINED = "quarantined"


class EvidenceRelation(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    CONTEXT = "context"


class RiskType(StrEnum):
    MISSED_DIAGNOSIS = "missed_diagnosis"
    DELAY_HARM = "delay_harm"
    TREATMENT_HARM = "treatment_harm"
    OTHER = "other"


class FirewallDecision(StrEnum):
    ACCEPT = "accept"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class ClaimProvenance:
    """Provenance assigned by the trusted orchestrator, never by an agent."""

    claim_id: str
    case_id: str
    origin_message_id: str
    origin_agent_id: str
    parent_claim_ids: tuple[str, ...] = ()
    exposed_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier("claim_id", self.claim_id)
        _require_identifier("case_id", self.case_id)
        _require_identifier("origin_message_id", self.origin_message_id)
        _require_identifier("origin_agent_id", self.origin_agent_id)
        object.__setattr__(
            self,
            "parent_claim_ids",
            _as_unique_tuple(
                self.parent_claim_ids,
                name="parent_claim_ids",
            ),
        )
        object.__setattr__(
            self,
            "exposed_claim_ids",
            _as_unique_tuple(
                self.exposed_claim_ids,
                name="exposed_claim_ids",
            ),
        )
        if self.claim_id in self.parent_claim_ids:
            raise SchemaValidationError("a claim cannot be its own parent")
        if self.claim_id in self.exposed_claim_ids:
            raise SchemaValidationError("a claim cannot expose itself")


@dataclass(frozen=True)
class CaseFact:
    """An immutable, addressable span from the original case text."""

    fact_id: str
    case_id: str
    text: str
    start_char: int
    end_char: int
    fact_type: FactType = FactType.OTHER
    polarity: Polarity = Polarity.PRESENT
    temporality: Temporality = Temporality.CURRENT
    modality: Modality = Modality.TEXT

    def __post_init__(self) -> None:
        _require_identifier("fact_id", self.fact_id)
        _require_identifier("case_id", self.case_id)
        _require_non_empty("text", self.text)
        _require_int("start_char", self.start_char)
        _require_int("end_char", self.end_char)
        _require_enum("fact_type", self.fact_type, FactType)
        _require_enum("polarity", self.polarity, Polarity)
        _require_enum("temporality", self.temporality, Temporality)
        _require_enum("modality", self.modality, Modality)
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise SchemaValidationError(
                f"invalid character span for {self.fact_id}: "
                f"[{self.start_char}, {self.end_char})"
            )


@dataclass(frozen=True)
class DiagnosisRef:
    """Canonical identity for a diagnosis candidate.

    ``concept_id`` and ``ontology`` are populated exclusively by a trusted
    :class:`~seal_med.ontology.OntologyMapper` after agent output is compiled.
    Agents cannot self-assert medical identity; ``label`` and
    ``canonical_label`` remain untrusted display text until a concept_id is
    bound by the orchestrator. The strict decoder therefore rejects these
    fields from agent payloads.
    """

    candidate_id: str
    label: str
    canonical_label: str
    source: CandidateSource = CandidateSource.AGENT_PROPOSED
    concept_id: str | None = None
    ontology: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("candidate_id", self.candidate_id)
        _require_non_empty("label", self.label)
        _require_non_empty("canonical_label", self.canonical_label)
        _require_enum("source", self.source, CandidateSource)
        if (self.concept_id is None) != (self.ontology is None):
            raise SchemaValidationError(
                "concept_id and ontology must be both set or both absent"
            )
        if self.concept_id is not None:
            _require_non_empty("concept_id", self.concept_id)
            _require_non_empty("ontology", self.ontology)


@dataclass(frozen=True)
class TypedClaim:
    """A typed claim and its complete provenance declaration.

    This dataclass checks only schema invariants. ClaimFirewall can establish
    structural/extractive source binding, but neither layer establishes
    clinical truth or medical entailment.
    """

    claim_id: str
    case_id: str
    claim_type: ClaimType
    value: str
    origin_message_id: str
    origin_agent_id: str
    source_fact_ids: tuple[str, ...] = ()
    external_source_ids: tuple[str, ...] = ()
    parent_claim_ids: tuple[str, ...] = ()
    exposed_claim_ids: tuple[str, ...] = ()
    candidate_id: str | None = None
    relation: EvidenceRelation | None = None
    polarity: Polarity = Polarity.PRESENT
    short_rationale: str = ""

    def __post_init__(self) -> None:
        _require_identifier("claim_id", self.claim_id)
        _require_identifier("case_id", self.case_id)
        _require_enum("claim_type", self.claim_type, ClaimType)
        _require_non_empty("value", self.value)
        _require_identifier("origin_message_id", self.origin_message_id)
        _require_identifier("origin_agent_id", self.origin_agent_id)
        object.__setattr__(
            self,
            "source_fact_ids",
            _as_unique_tuple(
                self.source_fact_ids,
                name="source_fact_ids",
            ),
        )
        object.__setattr__(
            self,
            "external_source_ids",
            _as_unique_tuple(
                self.external_source_ids,
                name="external_source_ids",
                identifiers=False,
            ),
        )
        object.__setattr__(
            self,
            "parent_claim_ids",
            _as_unique_tuple(
                self.parent_claim_ids,
                name="parent_claim_ids",
            ),
        )
        object.__setattr__(
            self,
            "exposed_claim_ids",
            _as_unique_tuple(
                self.exposed_claim_ids,
                name="exposed_claim_ids",
            ),
        )
        if self.candidate_id is not None:
            _require_identifier("candidate_id", self.candidate_id)
        if self.relation is not None:
            _require_enum("relation", self.relation, EvidenceRelation)
        _require_enum("polarity", self.polarity, Polarity)
        if not isinstance(self.short_rationale, str):
            raise SchemaValidationError("short_rationale must be a string")
        if self.claim_id in self.parent_claim_ids:
            raise SchemaValidationError("a claim cannot be its own parent")
        if self.claim_id in self.exposed_claim_ids:
            raise SchemaValidationError("a claim cannot expose itself")
        if (
            self.claim_type is ClaimType.DIAGNOSTIC_HYPOTHESIS
            and self.candidate_id is None
        ):
            raise SchemaValidationError(
                "diagnostic_hypothesis must reference a candidate_id"
            )


@dataclass(frozen=True)
class RiskConcern:
    """A clinical-safety concern, with no executable action payload."""

    concern_id: str
    case_id: str
    risk_type: RiskType
    risk_claim_id: str
    severity: int
    time_criticality: int
    irreversibility: int
    evidence_fact_ids: tuple[str, ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()
    diagnosis_ref: DiagnosisRef | None = None

    def __post_init__(self) -> None:
        _require_identifier("concern_id", self.concern_id)
        _require_identifier("case_id", self.case_id)
        _require_enum("risk_type", self.risk_type, RiskType)
        _require_identifier("risk_claim_id", self.risk_claim_id)
        object.__setattr__(
            self,
            "evidence_fact_ids",
            _as_unique_tuple(
                self.evidence_fact_ids,
                name="evidence_fact_ids",
            ),
        )
        object.__setattr__(
            self,
            "evidence_claim_ids",
            _as_unique_tuple(
                self.evidence_claim_ids,
                name="evidence_claim_ids",
            ),
        )
        if (
            self.diagnosis_ref is not None
            and not isinstance(self.diagnosis_ref, DiagnosisRef)
        ):
            raise SchemaValidationError(
                "diagnosis_ref must be a DiagnosisRef or None"
            )
        _require_int("severity", self.severity)
        _require_int("time_criticality", self.time_criticality)
        _require_int("irreversibility", self.irreversibility)
        if not 0 <= self.severity <= 4:
            raise SchemaValidationError("severity must be between 0 and 4")
        if not 0 <= self.time_criticality <= 3:
            raise SchemaValidationError(
                "time_criticality must be between 0 and 3"
            )
        if not 0 <= self.irreversibility <= 2:
            raise SchemaValidationError(
                "irreversibility must be between 0 and 2"
            )
        if (
            self.risk_type is RiskType.MISSED_DIAGNOSIS
            and self.diagnosis_ref is None
        ):
            raise SchemaValidationError(
                "missed_diagnosis risk must reference a diagnosis"
            )


@dataclass(frozen=True)
class VerifiedEvidence:
    """An assessment emitted by a trusted verifier adapter.

    ``relation_strength`` is negative for counter-evidence and positive for
    support. A scorer accepts it only after registration in an
    ``EvidenceRegistry`` and inclusion in a current-snapshot batch.
    """

    assessment_id: str
    verifier_id: str
    verifier_version: str
    candidate_id: str
    case_id: str
    claim_id: str
    relation_strength: int
    source_reliability: float = 1.0
    specificity: float = 1.0
    fact_id: str | None = None
    external_source_id: str | None = None
    evidence_dimension: str = ""
    essential_conflict: bool = False

    def __post_init__(self) -> None:
        _require_identifier("assessment_id", self.assessment_id)
        _require_identifier("verifier_id", self.verifier_id)
        _require_non_empty("verifier_version", self.verifier_version)
        _require_identifier("candidate_id", self.candidate_id)
        _require_identifier("case_id", self.case_id)
        _require_identifier("claim_id", self.claim_id)
        _require_int("relation_strength", self.relation_strength)
        if (self.fact_id is None) == (self.external_source_id is None):
            raise SchemaValidationError(
                "verified evidence must reference exactly one of "
                "fact_id or external_source_id"
            )
        if self.fact_id is not None:
            _require_identifier("fact_id", self.fact_id)
        if self.external_source_id is not None:
            _require_non_empty("external_source_id", self.external_source_id)
        if not isinstance(self.evidence_dimension, str):
            raise SchemaValidationError("evidence_dimension must be a string")
        if type(self.essential_conflict) is not bool:
            raise SchemaValidationError("essential_conflict must be a boolean")
        if not -3 <= self.relation_strength <= 3:
            raise SchemaValidationError(
                "relation_strength must be between -3 and 3"
            )
        if self.relation_strength == 0:
            raise SchemaValidationError("relation_strength must not be zero")
        _require_number("source_reliability", self.source_reliability)
        _require_number("specificity", self.specificity)
        if not 0.0 <= self.source_reliability <= 1.0:
            raise SchemaValidationError(
                "source_reliability must be between 0 and 1"
            )
        if not 0.0 <= self.specificity <= 1.0:
            raise SchemaValidationError("specificity must be between 0 and 1")

    @property
    def source_key(self) -> str:
        if self.fact_id is not None:
            return f"fact:{self.fact_id}"
        return f"external:{self.external_source_id}"


@dataclass(frozen=True)
class EvidenceBatch:
    """Registered verifier assessments bound to one ledger snapshot."""

    snapshot_id: str
    ledger_snapshot_id: str
    assessments: tuple[VerifiedEvidence, ...]

    def __post_init__(self) -> None:
        _require_non_empty("snapshot_id", self.snapshot_id)
        _require_non_empty("ledger_snapshot_id", self.ledger_snapshot_id)
        if not isinstance(self.assessments, tuple):
            raise SchemaValidationError("assessments must be a tuple")
        seen: set[str] = set()
        for assessment in self.assessments:
            if not isinstance(assessment, VerifiedEvidence):
                raise SchemaValidationError(
                    "assessments may contain VerifiedEvidence only"
                )
            if assessment.assessment_id in seen:
                raise SchemaValidationError(
                    "assessment IDs must be unique within a batch"
                )
            seen.add(assessment.assessment_id)


@dataclass(frozen=True)
class FirewallResult:
    decision: FirewallDecision
    claim_id: str
    reasons: tuple[str, ...] = ()
    duplicate_of: str | None = None


@dataclass(frozen=True)
class RollbackResult:
    source_claim_id: str
    deactivated_claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosisScore:
    snapshot_id: str
    artifact_id: str
    candidate: DiagnosisRef
    positive_evidence: float
    negative_evidence: float
    sufficiency: float
    balance: float
    coverage: float
    source_independence: float
    support_score: float
    contradiction_ratio: float
    has_unresolved_essential_conflict: bool
    eligible: bool
    # Share of the case's total support score held by this candidate. Unlike
    # ``support_score`` this is a competitive quantity, and it separates correct
    # from incorrect top-1 answers far better (AUROC 0.75 vs 0.53 on archived
    # runs) because what predicts correctness is how decisively one candidate
    # leads its rivals, not how much evidence it accumulated in absolute terms.
    relative_support: float = 0.0
    used_source_keys: tuple[str, ...] = ()
    ignored_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskScore:
    snapshot_id: str
    artifact_id: str
    concern_id: str
    risk_type: RiskType
    risk_claim_id: str
    diagnosis_ref: DiagnosisRef | None
    severity: int
    time_criticality: int
    irreversibility: int
    risk_score: float
    evidence_linked: bool
    safety_review_required: bool
    declared_evidence_fact_ids: tuple[str, ...] = ()
    live_evidence_claim_ids: tuple[str, ...] = ()
    live_evidence_assessment_ids: tuple[str, ...] = ()
    ignored_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DualTrackResult:
    snapshot_id: str
    artifact_id: str
    diagnosis_scores: tuple[DiagnosisScore, ...]
    risk_scores: tuple[RiskScore, ...]


@dataclass(frozen=True)
class ClaimRecord:
    """Immutable snapshot of ledger state around an immutable typed claim."""

    claim: TypedClaim
    status: ClaimStatus
    taint_state: TaintState = TaintState.CLEAN
    taint_sources: tuple[str, ...] = ()
    intrinsic_taint: bool = False
    duplicate_of: str | None = None
    reasons: tuple[str, ...] = ()
    history: tuple[str, ...] = ()

    @property
    def is_servable(self) -> bool:
        return (
            self.claim.claim_type is ClaimType.SOURCE_OBSERVATION
            and self.taint_state is TaintState.CLEAN
            and self.status in {
                ClaimStatus.SOURCE_BOUND,
                ClaimStatus.SUPPORTED,
            }
        )
