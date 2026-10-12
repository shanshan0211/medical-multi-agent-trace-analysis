"""Deterministic registries used by the SEAL-Med M1 core."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Mapping

from .exceptions import (
    DuplicateIdentifierError,
    SchemaValidationError,
    UnknownCandidateError,
)
from .schemas import DiagnosisRef


def _canonical_identity(label: str) -> str:
    normalized = unicodedata.normalize("NFKC", label).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


class CandidateRegistry(Mapping[str, DiagnosisRef]):
    """A diagnosis-only registry.

    Risk concerns and recommended actions are stored in their own typed
    structures and can never be inserted here accidentally.
    """

    def __init__(self, candidates: tuple[DiagnosisRef, ...] = ()) -> None:
        self._by_id: dict[str, DiagnosisRef] = {}
        self._by_canonical_label: dict[str, str] = {}
        self._by_concept: dict[tuple[str, str], str] = {}
        for candidate in candidates:
            self.add(candidate)

    def add(self, candidate: DiagnosisRef) -> DiagnosisRef:
        if not isinstance(candidate, DiagnosisRef):
            raise SchemaValidationError(
                "candidate registry accepts DiagnosisRef objects only"
            )
        existing = self._by_id.get(candidate.candidate_id)
        if existing is not None:
            if existing == candidate:
                return existing
            raise DuplicateIdentifierError(
                f"candidate_id {candidate.candidate_id!r} was reused with "
                "different content"
            )

        canonical_key = _canonical_identity(candidate.canonical_label)
        canonical_owner = self._by_canonical_label.get(canonical_key)
        if canonical_owner is not None:
            raise DuplicateIdentifierError(
                f"canonical diagnosis {candidate.canonical_label!r} is "
                f"already registered as {canonical_owner!r}"
            )

        concept_owner: str | None = None
        if candidate.concept_id is not None and candidate.ontology is not None:
            concept_key = (candidate.ontology, candidate.concept_id)
            concept_owner = self._by_concept.get(concept_key)
            if concept_owner is not None:
                raise DuplicateIdentifierError(
                    f"concept {candidate.concept_id!r} "
                    f"({candidate.ontology!r}) is already registered as "
                    f"{concept_owner!r}"
                )

        self._by_id[candidate.candidate_id] = candidate
        self._by_canonical_label[canonical_key] = candidate.candidate_id
        if concept_owner is None and candidate.concept_id is not None:
            concept_key = (candidate.ontology, candidate.concept_id)
            self._by_concept[concept_key] = candidate.candidate_id
        return candidate

    def __getitem__(self, candidate_id: str) -> DiagnosisRef:
        try:
            return self._by_id[candidate_id]
        except KeyError as exc:
            raise UnknownCandidateError(
                f"unknown candidate_id: {candidate_id}"
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_id)

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def candidates(self) -> tuple[DiagnosisRef, ...]:
        return tuple(self._by_id.values())

    def validate_ref(self, candidate_id: str) -> None:
        if candidate_id not in self._by_id:
            raise UnknownCandidateError(
                f"unknown candidate_id: {candidate_id}"
            )

    def resolve_by_concept(self, ontology: str, concept_id: str) -> DiagnosisRef | None:
        """Return the candidate bound to ``(ontology, concept_id)`` if any."""

        if not isinstance(ontology, str) or not ontology.strip():
            raise SchemaValidationError("ontology must be a non-empty string")
        if not isinstance(concept_id, str) or not concept_id.strip():
            raise SchemaValidationError("concept_id must be a non-empty string")
        candidate_id = self._by_concept.get((ontology, concept_id))
        if candidate_id is None:
            return None
        return self._by_id[candidate_id]
