"""Immutable-record claim ledger with deterministic taint propagation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import fields, replace
from enum import Enum

from .exceptions import (
    DuplicateIdentifierError,
    SchemaValidationError,
    StaleSnapshotError,
    UnknownClaimError,
)
from .facts import FactIndex
from .registry import CandidateRegistry
from .schemas import (
    ClaimRecord,
    ClaimStatus,
    RollbackResult,
    TaintState,
    TypedClaim,
)


def _canonical_dataclass_payload(instance: object) -> dict[str, object]:
    """Serialize every declared field so snapshots cannot omit new content."""

    def normalize(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        return value

    return {
        field.name: normalize(getattr(instance, field.name))
        for field in fields(instance)
    }


class ClaimLedger(Mapping[str, ClaimRecord]):
    """Append-only provenance graph with versioned mutable state.

    Claims are never physically removed. Dependencies are expressed only by
    ``parent_claim_ids`` and propagate taint. ``exposed_claim_ids`` are audit
    metadata and deliberately do not propagate taint.
    """

    def __init__(
        self,
        fact_index: FactIndex,
        candidates: CandidateRegistry,
    ) -> None:
        if not isinstance(fact_index, FactIndex):
            raise SchemaValidationError("fact_index must be a FactIndex")
        if not isinstance(candidates, CandidateRegistry):
            raise SchemaValidationError(
                "candidates must be a CandidateRegistry"
            )
        self._fact_index = fact_index
        self._candidates = candidates
        self._records: dict[str, ClaimRecord] = {}
        self._children: dict[str, set[str]] = {}
        self._fingerprints: dict[str, str] = {}
        self._revision = 0

    def __getitem__(self, claim_id: str) -> ClaimRecord:
        try:
            return self._records[claim_id]
        except KeyError as exc:
            raise UnknownClaimError(f"unknown claim_id: {claim_id}") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def fact_index(self) -> FactIndex:
        return self._fact_index

    @property
    def candidates(self) -> CandidateRegistry:
        return self._candidates

    @property
    def snapshot_id(self) -> str:
        payload = {
            "case_id": self.fact_index.case_id,
            "case_sha256": self.fact_index.source_sha256,
            "revision": self._revision,
            "facts": [
                {
                    "fact_id": fact.fact_id,
                    "case_id": fact.case_id,
                    "text": fact.text,
                    "start_char": fact.start_char,
                    "end_char": fact.end_char,
                    "fact_type": fact.fact_type.value,
                    "polarity": fact.polarity.value,
                    "temporality": fact.temporality.value,
                    "modality": fact.modality.value,
                }
                for fact in self.fact_index.facts
            ],
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "label": candidate.label,
                    "canonical_label": candidate.canonical_label,
                    "source": candidate.source.value,
                    "concept_id": candidate.concept_id,
                    "ontology": candidate.ontology,
                }
                for candidate in sorted(
                    self.candidates.candidates,
                    key=lambda item: item.candidate_id,
                )
            ],
            "claims": [
                {
                    "claim": _canonical_dataclass_payload(record.claim),
                    "status": record.status.value,
                    "taint_state": record.taint_state.value,
                    "taint_sources": record.taint_sources,
                    "intrinsic_taint": record.intrinsic_taint,
                    "duplicate_of": record.duplicate_of,
                    "reasons": record.reasons,
                    "history": record.history,
                }
                for _, record in sorted(self._records.items())
            ],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"r{self._revision}:{digest}"

    @property
    def claims(self) -> tuple[ClaimRecord, ...]:
        return tuple(self._records.values())

    def find_duplicate(self, fingerprint: str) -> str | None:
        return self._fingerprints.get(fingerprint)

    def _commit_validated_claim(
        self,
        claim: TypedClaim,
        *,
        status: ClaimStatus,
        fingerprint: str,
        quarantined: bool = False,
        reasons: tuple[str, ...] = (),
    ) -> ClaimRecord:
        if claim.claim_id in self._records:
            existing = self._records[claim.claim_id]
            if existing.claim != claim:
                raise DuplicateIdentifierError(
                    f"claim_id {claim.claim_id!r} was reused with "
                    "different content"
                )
            return existing

        for parent_id in claim.parent_claim_ids:
            if parent_id not in self._records:
                raise UnknownClaimError(f"unknown parent claim: {parent_id}")
        for exposed_id in claim.exposed_claim_ids:
            if exposed_id not in self._records:
                raise UnknownClaimError(f"unknown exposed claim: {exposed_id}")

        taint_state = TaintState.QUARANTINED if quarantined else TaintState.CLEAN
        taint_sources = (claim.claim_id,) if quarantined else ()
        history = (
            f"created:{status.value}",
            *(f"reason:{reason}" for reason in reasons),
        )
        record = ClaimRecord(
            claim=claim,
            status=status,
            taint_state=taint_state,
            taint_sources=taint_sources,
            intrinsic_taint=quarantined,
            reasons=reasons,
            history=history,
        )
        self._records[claim.claim_id] = record
        self._fingerprints[fingerprint] = claim.claim_id
        self._children.setdefault(claim.claim_id, set())
        for parent_id in claim.parent_claim_ids:
            self._children.setdefault(parent_id, set()).add(claim.claim_id)
        self._recompute_taint()
        self._revision += 1
        return self._records[claim.claim_id]

    def is_servable(self, claim_id: str) -> bool:
        """Return whether the complete dependency closure is live."""

        self[claim_id]
        memo: dict[str, bool] = {}

        def visit(current_id: str, active_path: set[str]) -> bool:
            if current_id in memo:
                return memo[current_id]
            if current_id in active_path:
                raise SchemaValidationError("claim graph contains a cycle")
            active_path.add(current_id)
            record = self._records[current_id]
            locally_servable = record.is_servable
            parents_servable = all(
                visit(parent_id, active_path)
                for parent_id in record.claim.parent_claim_ids
            )
            active_path.remove(current_id)
            memo[current_id] = locally_servable and parents_servable
            return memo[current_id]

        return visit(claim_id, set())

    def servable_claim_ids(self) -> tuple[str, ...]:
        return tuple(
            claim_id for claim_id in self._records if self.is_servable(claim_id)
        )

    def is_reviewable(self, claim_id: str) -> bool:
        """Return whether a proposal may enter a non-executable review queue."""

        record = self[claim_id]
        if record.taint_state is not TaintState.CLEAN:
            return False
        if record.status not in {
            ClaimStatus.PROPOSED,
            ClaimStatus.SOURCE_BOUND,
            ClaimStatus.SUPPORTED,
        }:
            return False
        return all(
            self.is_servable(parent_id)
            for parent_id in record.claim.parent_claim_ids
        )

    def assert_snapshot(self, snapshot_id: str) -> None:
        if snapshot_id != self.snapshot_id:
            raise StaleSnapshotError(
                f"artifact snapshot {snapshot_id!r} is stale; "
                f"current ledger snapshot is {self.snapshot_id!r}"
            )

    def mark_disputed(self, claim_id: str, reason: str) -> ClaimRecord:
        record = self[claim_id]
        if record.status in {ClaimStatus.REFUTED, ClaimStatus.EXPIRED}:
            return record
        if record.status is ClaimStatus.DISPUTED and reason in record.reasons:
            return record
        updated = replace(
            record,
            status=ClaimStatus.DISPUTED,
            reasons=record.reasons + (reason,),
            history=record.history + (f"disputed:{reason}",),
        )
        self._records[claim_id] = updated
        self._revision += 1
        return updated

    def mark_refuted(self, claim_id: str, reason: str) -> RollbackResult:
        record = self[claim_id]
        before = self._effective_active_map()
        changed = not (
            record.status is ClaimStatus.REFUTED
            and record.intrinsic_taint
            and reason in record.reasons
        )
        if changed:
            self._records[claim_id] = replace(
                record,
                status=ClaimStatus.REFUTED,
                intrinsic_taint=True,
                reasons=record.reasons + (reason,),
                history=record.history + (f"refuted:{reason}",),
            )
            self._recompute_taint()
            self._revision += 1
        return RollbackResult(
            source_claim_id=claim_id,
            deactivated_claim_ids=self._newly_deactivated(before),
        )

    def rollback(self, claim_id: str, reason: str) -> RollbackResult:
        """Quarantine a source and recursively deactivate dependent claims."""

        record = self[claim_id]
        before = self._effective_active_map()
        changed = not (
            record.intrinsic_taint
            and reason in record.reasons
        )
        if changed:
            self._records[claim_id] = replace(
                record,
                intrinsic_taint=True,
                taint_state=TaintState.QUARANTINED,
                taint_sources=(claim_id,),
                reasons=record.reasons + (reason,),
                history=record.history + (f"rollback:{reason}",),
            )
            self._recompute_taint()
            self._revision += 1
        return RollbackResult(
            source_claim_id=claim_id,
            deactivated_claim_ids=self._newly_deactivated(before),
        )

    def descendants_of(self, claim_id: str) -> tuple[str, ...]:
        self[claim_id]
        visited: set[str] = set()
        queue = list(self._children.get(claim_id, ()))
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            queue.extend(sorted(self._children.get(current_id, ())))
        return tuple(sorted(visited))

    def _effective_active_map(self) -> dict[str, bool]:
        return {
            claim_id: self.is_servable(claim_id)
            for claim_id in self._records
        }

    def _newly_deactivated(self, before: dict[str, bool]) -> tuple[str, ...]:
        return tuple(
            claim_id
            for claim_id in self._records
            if before.get(claim_id, False) and not self.is_servable(claim_id)
        )

    def _recompute_taint(self) -> None:
        """Propagate intrinsic taint over dependency edges to a fixed point."""

        for claim_id, record in tuple(self._records.items()):
            parent_sources: set[str] = set()
            for parent_id in record.claim.parent_claim_ids:
                parent_sources.update(self._records[parent_id].taint_sources)
            if record.intrinsic_taint:
                state = (
                    TaintState.TAINTED
                    if record.status is ClaimStatus.REFUTED
                    else TaintState.QUARANTINED
                )
                parent_sources.add(claim_id)
                sources = tuple(sorted(parent_sources))
            else:
                sources = tuple(sorted(parent_sources))
                state = TaintState.TAINTED if sources else TaintState.CLEAN
            if (
                state is not record.taint_state
                or sources != record.taint_sources
            ):
                self._records[claim_id] = replace(
                    record,
                    taint_state=state,
                    taint_sources=sources,
                    history=record.history
                    + (
                        f"taint:{state.value}:"
                        f"{','.join(sources) if sources else 'none'}",
                    ),
                )
