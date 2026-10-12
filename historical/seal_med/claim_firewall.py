"""Deterministic claim validation before ledger admission."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from .exceptions import (
    DuplicateIdentifierError,
    SchemaValidationError,
    UnknownCandidateError,
    UnknownClaimError,
    UnknownFactError,
)
from .ledger import ClaimLedger
from .schemas import (
    ClaimStatus,
    ClaimType,
    EvidenceRelation,
    FirewallDecision,
    FirewallResult,
    Polarity,
    TaintState,
    TypedClaim,
)


_NUMBER_TOKEN = (
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)
_NUMBER_PATTERN = re.compile(rf"(?<![\w.]){_NUMBER_TOKEN}")
_MEASUREMENT_PATTERN = re.compile(
    rf"(?<![\w.])({_NUMBER_TOKEN})\s*"
    r"([A-Za-zµμ°%]+(?:/[A-Za-z]+)?)\b",
    re.IGNORECASE,
)
# The pattern above accepts *any* word after a number, which made "0.3 to 1.2"
# yield the measurement (0.3, "to"), "Ki-67 index" yield (67, "index") and
# "S-100 revealed" yield (100, "revealed"). Since a value's measurements must be
# a subset of the source's, those phantom pairs quarantined faithful claims —
# 27.5% of the 5724 pairs extracted from the curated corpora had a unit that
# cannot be a unit under any reading, concentrated in the lab-heavy cases where
# numeric fidelity matters most.
#
# Units are therefore recognised explicitly. An unrecognised unit only drops the
# pair; the bare number must still appear in the source via ``_numbers``, and the
# content-overlap floor still has to be met, so protection degrades rather than
# disappears. Atoms are curated from the units actually present in
# ``data/curated_cases`` plus standard clinical units, and compound units are
# accepted when every "/"-separated part is an atom ("mg/dl", "beats/min").
_UNIT_ATOMS: frozenset[str] = frozenset(
    {
        # mass and amount
        "g", "mg", "kg", "mcg", "ug", "µg", "μg", "ng", "pg", "fg",
        "mol", "mmol", "μmol", "µmol", "umol", "nmol", "pmol", "fmol",
        "meq", "eq", "mosm", "osm", "kat", "μkat", "µkat", "ukat",
        "iu", "miu", "ku", "u", "au", "mu",
        # volume
        "l", "dl", "ml", "μl", "µl", "ul", "cl", "fl", "nl", "pl",
        # length and area. "inch"/"inches" are listed but bare "in" is not: as an
        # abbreviation it is indistinguishable from the preposition, which the
        # corpora produce as a phantom unit 113 times ("involvement in 3 lobes").
        "m", "cm", "mm", "μm", "µm", "um", "nm", "km", "dm",
        "inch", "inches", "ft", "feet", "foot", "lb", "lbs", "oz",
        # pressure
        "mmhg", "cmh", "kpa", "pa", "bar", "atm", "hu",
        # time
        "s", "sec", "secs", "second", "seconds", "ms", "millisecond",
        "milliseconds", "min", "mins", "minute", "minutes", "h", "hr", "hrs",
        "hour", "hours", "d", "day", "days", "week", "weeks", "month", "months",
        "year", "years", "decade", "decades",
        # rate and frequency
        "bpm", "hz", "khz", "mhz", "rpm", "db",
        # temperature and proportion
        "°c", "°f", "c", "f", "k", "%",
        # radiation and electricity
        "gy", "cgy", "mgy", "sv", "msv", "mv", "v", "mw", "w",
        # counted quantities that behave as units in clinical prose
        "beats", "beat", "breaths", "breath", "cells", "cell", "copies", "copy",
        "cycles", "cycle", "doses", "dose", "units", "unit", "times", "fold",
        "hpf", "lpf", "wbc", "rbc", "pack", "packs", "puffs", "tablets",
        "stools", "episodes", "leukocytes", "platelets",
    }
)


def _is_recognised_unit(unit: str) -> bool:
    """Accept a unit only if every "/"-separated part is a known atom."""

    parts = unit.split("/")
    return all(part in _UNIT_ATOMS for part in parts) if parts else False
_NEGATIVE_VALUE_PATTERN = re.compile(
    r"\b(?:no|not|without|denies?|negative|absent|absence|afebrile)\b"
    r"|(?:无|未见|未发现|否认|阴性|不伴)",
    re.IGNORECASE,
)
# Medical proper nouns where "negative" is a noun (blood type / stain), not a
# polarity marker. These are stripped from the text before polarity scanning
# so that "O-negative blood" does not flip a PRESENT observation to ABSENT.
# Covers ABO blood types (A/B/AB/O with optional Rh suffix), Rh factor alone,
# and Gram stain results.
_MEDICAL_PROPER_NOUN_STRIP = re.compile(
    r"\b(?:"
    r"(?:A|B|AB|O)[ -]?(?:positive|negative)"  # ABO blood types
    r"|Rh[ -]?(?:positive|negative)"             # Rh factor
    r"|gram[ -]?(?:positive|negative)"           # Gram stain
    r")\b",
    re.IGNORECASE,
)
_UNCERTAIN_VALUE_PATTERN = re.compile(
    r"\b(?:possible|possibly|probable|suspected|may|might|uncertain)\b"
    r"|(?:可能|考虑|疑似|待排|不确定)",
    re.IGNORECASE,
)
_LATERALITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "left": re.compile(r"\bleft\b|左(?:侧)?", re.IGNORECASE),
    "right": re.compile(r"\bright\b|右(?:侧)?", re.IGNORECASE),
    "bilateral": re.compile(r"\bbilateral\b|双侧", re.IGNORECASE),
}

_NON_DIAGNOSTIC_TYPES = {
    ClaimType.COMPLICATION,
    ClaimType.RISK_ITEM,
    ClaimType.RECOMMENDED_ACTION,
}


# NFKC already folds Unicode superscript digits into ASCII, so "10⁹/L" and
# "109/L" compare equal. Caret notation does not fold, so a claim written
# "2.3 × 10^3/μL" against a source written "2.3 × 103/μL" yields numeric sets
# {10, 3} versus {103} and the numeric guard rejects a faithful claim. Lab values
# are exactly where that guard matters most, so the exponent marker is collapsed
# to match the folded form.
_CARET_EXPONENT_PATTERN = re.compile(r"(?<=\d)\s*\^\s*(?=\d)")


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _CARET_EXPONENT_PATTERN.sub("", normalized)
    return " ".join(normalized.split())


def _is_extractively_bound(value: str, source: str) -> bool:
    """Check substring binding without accepting partial ASCII words."""

    start = source.find(value)
    while start >= 0:
        end = start + len(value)
        left_ok = not (
            value[0].isascii()
            and value[0].isalnum()
            and start > 0
            and source[start - 1].isascii()
            and source[start - 1].isalnum()
        )
        right_ok = not (
            value[-1].isascii()
            and value[-1].isalnum()
            and end < len(source)
            and source[end].isascii()
            and source[end].isalnum()
        )
        if left_ok and right_ok:
            return True
        start = source.find(value, start + 1)
    return False


# Minimum content-token overlap ratio for a source_observation to be considered
# a valid paraphrase of a source fact when strict extractive binding fails.
# Threshold calibrated to accept minor morphological/paraphrase variation
# ("feels hot" ↔ "feeling hot") while still rejecting unrelated text. Numbers
# and measurements must still extractively match to prevent hallucinated
# numeric content — only the textual tail around them is relaxed.
_PARAPHRASE_MIN_OVERLAP = 0.6
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "with", "and", "or", "to", "in", "on",
        "is", "are", "was", "were", "for", "by", "at", "as", "that", "this",
        "patient", "now", "complains", "complaint", "reports", "presents",
    }
)
# Trailing punctuation to strip when comparing content tokens so that
# "shivering." in the source matches "shivering" in the claim value.
_TRAILING_PUNCT = ".,;:!?)]}\"'"


def _stem_token(token: str) -> str:
    """Light English stemmer for common morphological variants.

    Handles plural (-s, -es, -ies→y) and verb inflections (-ing, -ed) so
    that "feels"/"feeling" reduce to a common stem. Pure ASCII, no locale
    dependency; deliberately conservative to avoid over-merging.
    """

    if not token or len(token) < 4:
        return token
    lower = token.casefold()
    for suffix, replacement in (
        ("ies", "y"),
        ("ing", ""),
        ("ied", "y"),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ):
        if lower.endswith(suffix) and len(lower) - len(suffix) >= 3:
            return lower[: -len(suffix)] + replacement
    return lower


def _content_tokens(text: str) -> list[str]:
    return [
        _stem_token(t.rstrip(_TRAILING_PUNCT).lstrip(_TRAILING_PUNCT))
        for t in text.split()
        if t and t not in _STOPWORDS
    ]


def _is_paraphrase_bound(value: str, source: str) -> bool:
    """Token-overlap fallback for minor morphological variation.

    Accepts paraphrases like "feels hot with shivering" against source text
    "feeling hot with shivering" while preserving zero-hallucination safety
    by requiring every numeric/measure token in the value to be present in
    the source (numbers cannot be invented by paraphrase).
    """

    value_tokens = _content_tokens(value)
    if not value_tokens:
        return _is_extractively_bound(value, source)
    source_tokens = set(_content_tokens(source))
    overlap = sum(1 for t in value_tokens if t in source_tokens)
    if overlap / len(value_tokens) < _PARAPHRASE_MIN_OVERLAP:
        return False
    # Numbers and measurements must still extractively match — paraphrase
    # cannot invent or alter numeric content.
    value_numbers = _numbers(value)
    if value_numbers and not value_numbers.issubset(_numbers(source)):
        return False
    value_measurements = _measurements(value)
    if value_measurements and not value_measurements.issubset(
        _measurements(source)
    ):
        return False
    value_lateralities = _lateralities(value)
    if value_lateralities and not value_lateralities.issubset(
        _lateralities(source)
    ):
        return False
    return True


def _fingerprint(claim: TypedClaim) -> str:
    payload = {
        "claim_type": claim.claim_type.value,
        "value": _normalize_text(claim.value),
        "source_fact_ids": sorted(claim.source_fact_ids),
        "external_source_ids": sorted(claim.external_source_ids),
        "parent_claim_ids": sorted(claim.parent_claim_ids),
        "candidate_id": claim.candidate_id,
        "relation": claim.relation.value if claim.relation else None,
        "polarity": claim.polarity.value,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _numbers(text: str) -> set[str]:
    return {
        match.group(0).replace(",", "").lstrip("+").lower()
        for match in _NUMBER_PATTERN.finditer(text)
    }


def _lateralities(text: str) -> set[str]:
    return {
        name
        for name, pattern in _LATERALITY_PATTERNS.items()
        if pattern.search(text)
    }


def _measurements(text: str) -> set[tuple[str, str]]:
    pairs = set()
    for match in _MEASUREMENT_PATTERN.finditer(text):
        unit = match.group(2).casefold()
        if not _is_recognised_unit(unit):
            continue
        pairs.add((match.group(1).replace(",", "").lstrip("+").lower(), unit))
    return pairs


def _value_polarity(text: str) -> Polarity:
    # Strip medical proper nouns (blood types, Gram stains) so that
    # "O-negative blood" does not trigger a false ABSENT polarity.
    stripped = _MEDICAL_PROPER_NOUN_STRIP.sub(" ", text)
    if _NEGATIVE_VALUE_PATTERN.search(stripped):
        return Polarity.ABSENT
    if _UNCERTAIN_VALUE_PATTERN.search(stripped):
        return Polarity.UNCERTAIN
    return Polarity.PRESENT


class ClaimFirewall:
    """Fail-closed validation for typed claims.

    Passing this firewall means the claim is structurally source-bound. It does
    not mean that the cited evidence medically entails the claim.
    """

    def __init__(self, ledger: ClaimLedger, *, disabled: bool = False) -> None:
        self.ledger = ledger
        self._disabled = bool(disabled)

    @property
    def disabled(self) -> bool:
        """``True`` when source-binding checks are bypassed (ablation arm)."""
        return self._disabled

    def submit(self, claim: TypedClaim) -> FirewallResult:
        if not isinstance(claim, TypedClaim):
            raise SchemaValidationError(
                "claim firewall accepts TypedClaim objects only"
            )
        if claim.claim_id in self.ledger:
            existing = self.ledger[claim.claim_id]
            if existing.claim == claim:
                blocked = (
                    existing.taint_state is not TaintState.CLEAN
                    or existing.status
                    in {
                        ClaimStatus.DISPUTED,
                        ClaimStatus.REFUTED,
                        ClaimStatus.EXPIRED,
                        ClaimStatus.UNKNOWN,
                    }
                )
                return FirewallResult(
                    decision=(
                        FirewallDecision.QUARANTINE
                        if blocked
                        else FirewallDecision.ACCEPT
                    ),
                    claim_id=claim.claim_id,
                    reasons=(
                        "idempotent_replay",
                        f"current_status:{existing.status.value}",
                        f"current_taint:{existing.taint_state.value}",
                    ),
                )
            raise DuplicateIdentifierError(
                f"claim_id {claim.claim_id!r} was reused with different content"
            )

        # Source-binding checks are the structural safety core of the firewall.
        # The ablation arm disables ONLY this check; type/duplicate/quarantine
        # guards remain active so the ledger stays internally consistent.
        reference_errors: list[str] = []
        if not self._disabled:
            reference_errors = self._reference_errors(claim)
            if reference_errors:
                return FirewallResult(
                    decision=FirewallDecision.REJECT,
                    claim_id=claim.claim_id,
                    reasons=tuple(reference_errors),
                )

        type_errors = self._type_errors(claim)
        if type_errors:
            return FirewallResult(
                decision=FirewallDecision.REJECT,
                claim_id=claim.claim_id,
                reasons=tuple(type_errors),
            )

        fingerprint = _fingerprint(claim)
        duplicate_of = self.ledger.find_duplicate(fingerprint)
        if duplicate_of is not None:
            return FirewallResult(
                decision=FirewallDecision.DUPLICATE,
                claim_id=claim.claim_id,
                reasons=("duplicate_evidence_does_not_add_weight",),
                duplicate_of=duplicate_of,
            )

        quarantine_reasons = self._quarantine_reasons(claim)
        status = self._initial_status(claim)
        record = self.ledger._commit_validated_claim(
            claim,
            status=status,
            fingerprint=fingerprint,
            quarantined=bool(quarantine_reasons),
            reasons=tuple(quarantine_reasons),
        )
        inherited_taint_reasons = [
            f"depends_on_tainted_claim:{parent_id}"
            for parent_id in claim.parent_claim_ids
            if self.ledger[parent_id].taint_sources
        ]
        result_reasons = tuple(quarantine_reasons + inherited_taint_reasons)
        return FirewallResult(
            decision=(
                FirewallDecision.QUARANTINE
                if (
                    quarantine_reasons
                    or record.taint_state is not TaintState.CLEAN
                )
                else FirewallDecision.ACCEPT
            ),
            claim_id=claim.claim_id,
            reasons=result_reasons,
        )

    def _reference_errors(self, claim: TypedClaim) -> list[str]:
        errors: list[str] = []
        if claim.case_id != self.ledger.fact_index.case_id:
            errors.append(
                f"claim case_id {claim.case_id!r} does not match active "
                f"case {self.ledger.fact_index.case_id!r}"
            )
        try:
            self.ledger.fact_index.validate_refs(claim.source_fact_ids)
        except UnknownFactError as exc:
            errors.append(str(exc))

        if claim.candidate_id is not None:
            try:
                self.ledger.candidates.validate_ref(claim.candidate_id)
            except UnknownCandidateError as exc:
                errors.append(str(exc))

        for parent_id in claim.parent_claim_ids:
            try:
                self.ledger[parent_id]
            except UnknownClaimError as exc:
                errors.append(str(exc))
        for exposed_id in claim.exposed_claim_ids:
            try:
                self.ledger[exposed_id]
            except UnknownClaimError as exc:
                errors.append(str(exc))
        return errors

    @staticmethod
    def _type_errors(claim: TypedClaim) -> list[str]:
        errors: list[str] = []
        if claim.relation is not None and claim.candidate_id is None:
            errors.append("evidence relation requires candidate_id")
        if (
            claim.claim_type in _NON_DIAGNOSTIC_TYPES
            and claim.relation
            in {EvidenceRelation.SUPPORT, EvidenceRelation.CONTRADICT}
        ):
            errors.append(
                "risk, complication and action claims cannot contribute "
                "diagnostic evidence"
            )
        if (
            claim.claim_type is ClaimType.DIAGNOSTIC_HYPOTHESIS
            and claim.relation is not None
        ):
            errors.append(
                "a diagnostic hypothesis is a candidate, not evidence for itself"
            )
        if (
            claim.claim_type is ClaimType.SOURCE_OBSERVATION
            and claim.external_source_ids
        ):
            errors.append(
                "source_observation cannot declare external sources; "
                "use a medical_rule claim"
            )
        return errors

    def _quarantine_reasons(self, claim: TypedClaim) -> list[str]:
        reasons: list[str] = []
        if (
            claim.claim_type is ClaimType.SOURCE_OBSERVATION
            and not claim.source_fact_ids
        ):
            reasons.append("source_observation_requires_fact_reference")

        if claim.claim_type is ClaimType.MEDICAL_RULE:
            reasons.append("external_medical_rule_requires_future_verification")

        if claim.claim_type is ClaimType.SOURCE_OBSERVATION:
            referenced = [
                self.ledger.fact_index[fact_id]
                for fact_id in claim.source_fact_ids
            ]
            source_text = " ".join(fact.text for fact in referenced)
            normalized_value = _normalize_text(claim.value)
            if referenced and not any(
                _is_extractively_bound(
                    normalized_value,
                    _normalize_text(fact.text),
                )
                for fact in referenced
            ):
                # Paraphrase fallback: accept minor morphological variation
                # ("feels hot" ↔ "feeling hot") when strict extractive
                # substring binding fails but content-token overlap is high
                # and all numeric/measure tokens still match the source.
                # This recovers valid evidence that the strict matcher was
                # dropping while preserving zero-hallucination safety
                # (numbers, lateralities, and measurements are still
                # required to extractively match).
                if not any(
                    _is_paraphrase_bound(
                        normalized_value,
                        _normalize_text(fact.text),
                    )
                    for fact in referenced
                ):
                    reasons.append(
                        "source_observation_must_be_extractively_bound_to_one_fact"
                    )
            if _value_polarity(claim.value) is not claim.polarity:
                reasons.append(
                    "declared_polarity_not_expressed_in_observation_value"
                )
            novel_numbers = _numbers(claim.value) - _numbers(source_text)
            if novel_numbers:
                reasons.append(
                    "observation_contains_numbers_absent_from_source:"
                    + ",".join(sorted(novel_numbers))
                )

            claim_lateralities = _lateralities(claim.value)
            source_lateralities = _lateralities(source_text)
            if claim_lateralities - source_lateralities:
                reasons.append(
                    "observation_introduces_unsupported_laterality:"
                    + ",".join(sorted(claim_lateralities - source_lateralities))
                )

            novel_measurements = _measurements(claim.value) - _measurements(
                source_text
            )
            if novel_measurements:
                formatted = ",".join(
                    f"{number}{unit}"
                    for number, unit in sorted(novel_measurements)
                )
                reasons.append(
                    "observation_contains_measurement_absent_from_source:"
                    + formatted
                )

            protected_numbers = _numbers(claim.value)
            protected_lateralities = _lateralities(claim.value)
            protected_measurements = _measurements(claim.value)
            if (
                referenced
                and (
                    protected_numbers
                    or protected_lateralities
                    or protected_measurements
                )
                and not any(
                    protected_numbers.issubset(_numbers(fact.text))
                    and protected_lateralities.issubset(
                        _lateralities(fact.text)
                    )
                    and protected_measurements.issubset(
                        _measurements(fact.text)
                    )
                    for fact in referenced
                )
            ):
                reasons.append(
                    "protected_atoms_do_not_cooccur_in_one_source_fact"
                )

            source_polarities = {fact.polarity for fact in referenced}
            if referenced and claim.polarity not in source_polarities:
                reasons.append("observation_polarity_conflicts_with_source")

        for exposed_id in claim.exposed_claim_ids:
            exposed = self.ledger[exposed_id]
            exposed_texts = tuple(
                normalized
                for normalized in (
                    _normalize_text(exposed.claim.value),
                    _normalize_text(exposed.claim.short_rationale),
                )
                if normalized
            )
            current_texts = tuple(
                normalized
                for normalized in (
                    _normalize_text(claim.value),
                    _normalize_text(claim.short_rationale),
                )
                if normalized
            )
            if exposed_id not in claim.parent_claim_ids and any(
                _is_extractively_bound(exposed_text, current_text)
                for exposed_text in exposed_texts
                for current_text in current_texts
            ):
                reasons.append(
                    f"undeclared_exact_adoption_of_exposed_claim:{exposed_id}"
                )

        return reasons

    @staticmethod
    def _initial_status(claim: TypedClaim) -> ClaimStatus:
        if claim.claim_type is ClaimType.MEDICAL_RULE:
            return ClaimStatus.UNKNOWN
        if (
            claim.claim_type is ClaimType.SOURCE_OBSERVATION
            and claim.source_fact_ids
        ):
            return ClaimStatus.SOURCE_BOUND
        return ClaimStatus.PROPOSED
