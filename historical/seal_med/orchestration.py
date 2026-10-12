"""Typed contracts shared by the SEAL-Med M2/M3 orchestration layers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .exceptions import SchemaValidationError
from .schemas import EvidenceRelation, RiskType


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SchemaValidationError(f"{name} is not a valid identifier: {value!r}")
    return value


def _text(name: str, value: object, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        raise SchemaValidationError(f"{name} must be a string{'' if allow_blank else ' and not blank'}")
    return value.strip()


def _string_tuple(name: str, value: object, *, limit: int | None = None) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SchemaValidationError(f"{name} must be an array of strings")
    result: list[str] = []
    for item in value:
        result.append(_text(name, item))
    result = list(dict.fromkeys(result))
    if limit is not None and len(result) > limit:
        raise SchemaValidationError(f"{name} may contain at most {limit} items")
    return tuple(result)


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SchemaValidationError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _number(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise SchemaValidationError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _strict(payload: object, required: set[str], optional: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SchemaValidationError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise SchemaValidationError(f"{name} field names must be strings")
    keys = set(payload)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise SchemaValidationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise SchemaValidationError(f"{name} contains unknown fields: {sorted(unknown)}")
    return payload


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Stage(StrEnum):
    INGEST = "ingest"
    ROUTE = "route"
    SEALED_PROPOSE = "sealed_propose"
    GROUND_AND_CANONICALIZE = "ground_and_canonicalize"
    MONITOR = "monitor"
    TRIAGE = "triage"
    TARGETED_INTERVENTION = "targeted_intervention"
    COMMON_MODE_CHECK = "common_mode_check"
    BLIND_VERIFY = "blind_verify"
    SCORE = "score"
    FINALIZE = "finalize"
    PROVISIONAL = "provisional"
    ESCALATE = "escalate"
    FAILED = "failed"


class AgentRole(StrEnum):
    INDEPENDENT_DIAGNOSTICIAN = "independent_diagnostician"
    MECHANISM_CHRONOLOGY_ANALYST = "mechanism_chronology_analyst"
    FALSIFIER = "falsifier"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class EventStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


class DetectorType(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM_AUDITOR = "llm_auditor"
    HYBRID = "hybrid"


class PolicyAction(StrEnum):
    QUARANTINE_AND_ROLLBACK = "quarantine_and_rollback"
    MINORITY_RESCUE = "minority_rescue"
    TARGETED_REVIEW = "targeted_review"
    VERIFY_EVIDENCE = "verify_evidence"
    COMMON_MODE_CHECK = "common_mode_check"
    PROCEED_TO_FINAL_VERIFY = "proceed_to_final_verify"
    STOP_NO_PROGRESS = "stop_no_progress"
    ESCALATE = "escalate"
    BUDGET_EXHAUSTED = "budget_exhausted"


class FinalStatus(StrEnum):
    CONDITIONAL_FINAL = "conditional_final"
    PROVISIONAL_DIFFERENTIAL = "provisional_differential"
    ABSTAIN_AND_ESCALATE = "abstain_and_escalate"
    FAIL_CLOSED = "fail_closed"


# A closed-set benchmark may present its entire label space per case. DDXPlus
# has 49 classes, so a limit of 20 would reject the aligned closed-set arm.
MAX_DIAGNOSIS_OPTIONS: int = 64


@dataclass(frozen=True)
class TaskRoute:
    task_type: str
    evidence_dimensions: tuple[str, ...]
    required_modalities: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    high_risk: bool
    diagnosis_options: tuple[str, ...] = ()
    max_targeted_reviews: int = 1

    def __post_init__(self) -> None:
        _text("task_type", self.task_type)
        object.__setattr__(self, "evidence_dimensions", _string_tuple("evidence_dimensions", self.evidence_dimensions))
        object.__setattr__(self, "required_modalities", _string_tuple("required_modalities", self.required_modalities))
        object.__setattr__(self, "required_capabilities", _string_tuple("required_capabilities", self.required_capabilities))
        object.__setattr__(
            self,
            "diagnosis_options",
            _string_tuple("diagnosis_options", self.diagnosis_options, limit=MAX_DIAGNOSIS_OPTIONS),
        )
        if type(self.high_risk) is not bool:
            raise SchemaValidationError("high_risk must be boolean")
        _integer("max_targeted_reviews", self.max_targeted_reviews, 0, 10)


@dataclass(frozen=True)
class EvidenceDraft:
    fact_id: str
    relation: EvidenceRelation
    observation: str
    short_rationale: str = ""

    def __post_init__(self) -> None:
        _identifier("fact_id", self.fact_id)
        if not isinstance(self.relation, EvidenceRelation):
            raise SchemaValidationError("relation must be EvidenceRelation")
        _text("observation", self.observation)
        _text("short_rationale", self.short_rationale, allow_blank=True)


@dataclass(frozen=True)
class CandidateDraft:
    label: str
    canonical_label: str
    is_primary: bool
    evidence: tuple[EvidenceDraft, ...]

    def __post_init__(self) -> None:
        _text("label", self.label)
        _text("canonical_label", self.canonical_label)
        if type(self.is_primary) is not bool:
            raise SchemaValidationError("is_primary must be boolean")
        if not isinstance(self.evidence, tuple) or any(not isinstance(item, EvidenceDraft) for item in self.evidence):
            raise SchemaValidationError("evidence must be a tuple of EvidenceDraft")
        support = sum(item.relation is EvidenceRelation.SUPPORT for item in self.evidence)
        contradict = sum(item.relation is EvidenceRelation.CONTRADICT for item in self.evidence)
        if support > 4 or contradict > 3:
            raise SchemaValidationError(
                "each candidate allows at most 4 support and 3 contradiction items; "
                f"candidate {self.label!r} returned {support} support and {contradict} contradiction items; "
                "drop the weakest items instead of resending the same evidence list"
            )


@dataclass(frozen=True)
class RiskDraft:
    diagnosis_label: str
    risk_type: RiskType
    severity: int
    time_criticality: int
    irreversibility: int
    evidence_fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("diagnosis_label", self.diagnosis_label)
        if not isinstance(self.risk_type, RiskType):
            raise SchemaValidationError("risk_type must be RiskType")
        _integer("severity", self.severity, 0, 4)
        _integer("time_criticality", self.time_criticality, 0, 3)
        _integer("irreversibility", self.irreversibility, 0, 2)
        facts = _string_tuple("evidence_fact_ids", self.evidence_fact_ids)
        for fact_id in facts:
            _identifier("evidence_fact_id", fact_id)
        object.__setattr__(self, "evidence_fact_ids", facts)


@dataclass(frozen=True)
class SealedProposal:
    message_id: str
    agent_id: str
    role: AgentRole
    model_id: str
    candidates: tuple[CandidateDraft, ...]
    unexplained_fact_ids: tuple[str, ...]
    uncertainty: tuple[str, ...]
    confidence: ConfidenceLevel

    def __post_init__(self) -> None:
        _identifier("message_id", self.message_id)
        _identifier("agent_id", self.agent_id)
        if not isinstance(self.role, AgentRole):
            raise SchemaValidationError("role must be AgentRole")
        _text("model_id", self.model_id)
        if not isinstance(self.candidates, tuple) or not 1 <= len(self.candidates) <= 3:
            raise SchemaValidationError("proposal must contain one to three candidates")
        if any(not isinstance(item, CandidateDraft) for item in self.candidates):
            raise SchemaValidationError("candidates must contain CandidateDraft objects")
        if sum(item.is_primary for item in self.candidates) != 1:
            raise SchemaValidationError("proposal must declare exactly one primary candidate")
        object.__setattr__(self, "unexplained_fact_ids", _string_tuple("unexplained_fact_ids", self.unexplained_fact_ids))
        object.__setattr__(self, "uncertainty", _string_tuple("uncertainty", self.uncertainty, limit=5))
        if not isinstance(self.confidence, ConfidenceLevel):
            raise SchemaValidationError("confidence must be ConfidenceLevel")

    @property
    def primary(self) -> CandidateDraft:
        return next(candidate for candidate in self.candidates if candidate.is_primary)


@dataclass(frozen=True)
class AgentEnvelope:
    proposal: SealedProposal
    risks: tuple[RiskDraft, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, SealedProposal):
            raise SchemaValidationError("proposal must be SealedProposal")
        if not isinstance(self.risks, tuple) or any(not isinstance(item, RiskDraft) for item in self.risks):
            raise SchemaValidationError("risks must be a tuple of RiskDraft")


@dataclass(frozen=True)
class TaxonomyEvent:
    event_id: str
    code: str
    stage: Stage
    severity: EventSeverity
    subject_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    detector: DetectorType
    confidence: float
    status: EventStatus
    policy_action: PolicyAction
    detail: str = ""

    def __post_init__(self) -> None:
        _identifier("event_id", self.event_id)
        _identifier("code", self.code)
        if not isinstance(self.stage, Stage) or not isinstance(self.severity, EventSeverity):
            raise SchemaValidationError("event stage/severity has invalid type")
        if not isinstance(self.detector, DetectorType) or not isinstance(self.status, EventStatus):
            raise SchemaValidationError("event detector/status has invalid type")
        if not isinstance(self.policy_action, PolicyAction):
            raise SchemaValidationError("policy_action has invalid type")
        object.__setattr__(self, "subject_ids", _string_tuple("subject_ids", self.subject_ids))
        object.__setattr__(self, "fact_ids", _string_tuple("fact_ids", self.fact_ids))
        _number("confidence", self.confidence, 0.0, 1.0)
        _text("detail", self.detail, allow_blank=True)


@dataclass(frozen=True)
class BudgetLimits:
    max_targeted_reviews: int = 1
    max_recruited_agents: int = 1
    max_schema_retries: int = 2
    max_llm_calls_per_case: int = 7
    max_total_tokens_per_case: int = 120_000
    max_intervention_rounds: int = 2
    max_error_exposures: int = 24

    def __post_init__(self) -> None:
        _integer("max_targeted_reviews", self.max_targeted_reviews, 0, 20)
        _integer("max_recruited_agents", self.max_recruited_agents, 0, 20)
        _integer("max_schema_retries", self.max_schema_retries, 0, 10)
        _integer("max_llm_calls_per_case", self.max_llm_calls_per_case, 1, 100)
        _integer("max_total_tokens_per_case", self.max_total_tokens_per_case, 1, 10_000_000)
        _integer("max_intervention_rounds", self.max_intervention_rounds, 0, 20)
        _integer("max_error_exposures", self.max_error_exposures, 0, 10_000)


@dataclass(frozen=True)
class CallAudit:
    call_id: str
    stage: Stage
    model_id: str
    prompt_hash: str
    response_hash: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    schema_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier("call_id", self.call_id)
        if not isinstance(self.stage, Stage):
            raise SchemaValidationError("stage must be Stage")
        _text("model_id", self.model_id)
        _text("prompt_hash", self.prompt_hash)
        _text("response_hash", self.response_hash)
        _integer("latency_ms", self.latency_ms, 0, 86_400_000)
        for name, value in (("input_tokens", self.input_tokens), ("output_tokens", self.output_tokens)):
            if value is not None and (type(value) is not int or value < 0):
                raise SchemaValidationError(f"{name} must be a non-negative integer or None")
        object.__setattr__(self, "schema_errors", _string_tuple("schema_errors", self.schema_errors))


@dataclass(frozen=True)
class StateTransition:
    stage: Stage
    reason: str
    ledger_snapshot_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise SchemaValidationError("stage must be Stage")
        _text("reason", self.reason)
        _text("ledger_snapshot_id", self.ledger_snapshot_id, allow_blank=True)


def decode_agent_envelope(
    payload: object,
    *,
    message_id: str,
    agent_id: str,
    role: AgentRole,
    model_id: str,
) -> AgentEnvelope:
    """Strictly decode an untrusted agent payload; provenance is injected here."""

    data = _strict(
        payload,
        {"candidates", "unexplained_fact_ids", "uncertainty", "confidence", "risks"},
        set(),
        "agent response",
    )
    raw_candidates = data["candidates"]
    if not isinstance(raw_candidates, list):
        raise SchemaValidationError("candidates must be an array")
    candidates: list[CandidateDraft] = []
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _strict(
            raw_candidate,
            {"label", "canonical_label", "is_primary", "evidence"},
            set(),
            f"candidate[{index}]",
        )
        if type(candidate["is_primary"]) is not bool:
            raise SchemaValidationError("is_primary must be boolean")
        raw_evidence = candidate["evidence"]
        if not isinstance(raw_evidence, list):
            raise SchemaValidationError("candidate evidence must be an array")
        evidence: list[EvidenceDraft] = []
        for evidence_index, raw_item in enumerate(raw_evidence):
            item = _strict(
                raw_item,
                {"fact_id", "relation", "observation", "short_rationale"},
                set(),
                f"evidence[{evidence_index}]",
            )
            try:
                relation = EvidenceRelation(_text("relation", item["relation"]))
            except ValueError as exc:
                raise SchemaValidationError("relation must be support, contradict, or context") from exc
            evidence.append(
                EvidenceDraft(
                    fact_id=_text("fact_id", item["fact_id"]),
                    relation=relation,
                    observation=_text("observation", item["observation"]),
                    short_rationale=_text("short_rationale", item["short_rationale"], allow_blank=True),
                )
            )
        candidates.append(
            CandidateDraft(
                label=_text("label", candidate["label"]),
                canonical_label=_text("canonical_label", candidate["canonical_label"]),
                is_primary=candidate["is_primary"],
                evidence=tuple(evidence),
            )
        )

    raw_risks = data["risks"]
    if not isinstance(raw_risks, list):
        raise SchemaValidationError("risks must be an array")
    risks: list[RiskDraft] = []
    for index, raw_risk in enumerate(raw_risks):
        risk = _strict(
            raw_risk,
            {"diagnosis_label", "risk_type", "severity", "time_criticality", "irreversibility", "evidence_fact_ids"},
            set(),
            f"risk[{index}]",
        )
        try:
            risk_type = RiskType(_text("risk_type", risk["risk_type"]))
        except ValueError as exc:
            raise SchemaValidationError("invalid risk_type") from exc
        risks.append(
            RiskDraft(
                diagnosis_label=_text("diagnosis_label", risk["diagnosis_label"]),
                risk_type=risk_type,
                severity=_integer("severity", risk["severity"], 0, 4),
                time_criticality=_integer("time_criticality", risk["time_criticality"], 0, 3),
                irreversibility=_integer("irreversibility", risk["irreversibility"], 0, 2),
                evidence_fact_ids=_string_tuple("evidence_fact_ids", risk["evidence_fact_ids"]),
            )
        )
    try:
        confidence = ConfidenceLevel(_text("confidence", data["confidence"]))
    except ValueError as exc:
        raise SchemaValidationError("confidence must be low, medium, or high") from exc
    proposal = SealedProposal(
        message_id=message_id,
        agent_id=agent_id,
        role=role,
        model_id=model_id,
        candidates=tuple(candidates),
        unexplained_fact_ids=_string_tuple("unexplained_fact_ids", data["unexplained_fact_ids"]),
        uncertainty=_string_tuple("uncertainty", data["uncertainty"], limit=5),
        confidence=confidence,
    )
    return AgentEnvelope(proposal=proposal, risks=tuple(risks))
