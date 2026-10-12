"""Deterministic case segmentation and immutable Fact Index construction."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping

from .exceptions import DuplicateIdentifierError, SchemaValidationError, UnknownFactError
from .schemas import CaseFact, FactType, Modality, Polarity, Temporality


_NEGATIVE_PATTERN = re.compile(
    r"\b(?:no|not|without|denies?|negative\s+for|absence\s+of|afebrile)\b"
    r"|(?:无|未见|未发现|否认|阴性|不伴)",
    re.IGNORECASE,
)
_UNCERTAIN_PATTERN = re.compile(
    r"\b(?:possible|possibly|probable|suspected|may|might|uncertain)\b"
    r"|(?:可能|考虑|疑似|待排|不确定)",
    re.IGNORECASE,
)
_PAST_PATTERN = re.compile(
    r"\b(?:history\s+of|previously|prior|past|formerly)\b"
    r"|(?:既往|曾经|此前|病史)",
    re.IGNORECASE,
)
_FUTURE_PATTERN = re.compile(
    r"\b(?:planned|will|scheduled|recommend(?:ed)?)\b"
    r"|(?:计划|拟行|建议|将行)",
    re.IGNORECASE,
)

_TYPE_PATTERNS: tuple[tuple[FactType, re.Pattern[str]], ...] = (
    (
        FactType.PATHOLOGY,
        re.compile(
            r"\b(?:patholog|histolog|biopsy|immunohistochem|ki-?67|cytolog)\w*\b"
            r"|(?:病理|活检|免疫组化|组织学|细胞学)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.IMAGING,
        re.compile(
            r"\b(?:mri|ct|pet(?:-ct)?|x-?ray|ultrasound|radiograph|imaging)\b"
            r"|(?:影像|超声|磁共振|放射|摄片)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.LABORATORY,
        re.compile(
            r"\b(?:laboratory|haemoglobin|hemoglobin|wbc|platelet|creatinine|"
            r"bilirubin|electrolyte|serology|assay)\b"
            r"|(?:实验室|化验|血常规|白细胞|血红蛋白|肌酐|胆红素|电解质)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.VITAL,
        re.compile(
            r"\b(?:blood\s+pressure|heart\s+rate|pulse|temperature|"
            r"respiratory\s+rate|spo2|oxygen\s+saturation)\b"
            r"|(?:血压|心率|脉搏|体温|呼吸频率|血氧)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.PHYSICAL_EXAM,
        re.compile(
            r"\b(?:physical\s+exam|examination|palpation|auscultation)\b"
            r"|(?:查体|体格检查|触诊|听诊)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.MEDICATION,
        re.compile(
            r"\b(?:medication|treated\s+with|therapy|dose|mg|antibiotic)\b"
            r"|(?:用药|治疗|剂量|抗生素|服用)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.DEMOGRAPHIC,
        re.compile(
            r"\b(?:\d{1,3}[- ]year[- ]old|male|female|man|woman|boy|girl)\b"
            r"|(?:\d{1,3}岁|男性|女性|男|女)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.HISTORY,
        re.compile(
            r"\b(?:history|for\s+\d+\s+(?:day|week|month|year)s?|since)\b"
            r"|(?:病史|持续|余年|余月|余天|以来)",
            re.IGNORECASE,
        ),
    ),
    (
        FactType.SYMPTOM,
        re.compile(
            r"\b(?:pain|fever|cough|fatigue|nausea|vomiting|dyspnea|rash)\b"
            r"|(?:疼痛|发热|咳嗽|乏力|恶心|呕吐|呼吸困难|皮疹)",
            re.IGNORECASE,
        ),
    ),
)

_ASSERTION_WORDS_PATTERN = re.compile(
    r"\b(?:no|not|without|denies?|negative|absence|absent|"
    r"present|presented|detected|identified|is|was|were|has|had|"
    r"but|however|while|whereas)\b"
    r"|(?:无|未见|未发现|否认|阴性|阳性|存在|出现|但|然而|而|同时)",
    re.IGNORECASE,
)
_CLAUSE_CONJUNCTION_PATTERN = re.compile(
    r"(?:but|however|while|whereas)\b|(?:但|然而|同时)",
    re.IGNORECASE,
)
_STRUCTURED_FIELD_PATTERN = re.compile(
    r"\s*['\"][^'\"]{1,80}['\"]\s*:",
    re.IGNORECASE,
)
_EXPLICIT_POSITIVE_PATTERN = re.compile(
    r"\b(?:present|positive|detected|identified|showed|revealed)\b"
    r"|(?:阳性|存在|可见)|\(\+\)",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = "\n。！？!?；;"


def _is_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char in _TERMINAL_PUNCTUATION:
        return True
    if char != ".":
        return False
    if index > 0 and index + 1 < len(text):
        if text[index - 1].isdigit() and text[index + 1].isdigit():
            return False
    return index + 1 == len(text) or text[index + 1].isspace()


def _is_soft_clause_boundary(text: str, index: int, start: int) -> bool:
    """Split only auditable comma transitions, not ordinary negated lists."""

    if text[index] not in {",", "，"}:
        return False
    if index > 0 and index + 1 < len(text):
        if text[index - 1].isdigit() and text[index + 1].isdigit():
            return False
    right_end = index + 1
    while right_end < len(text) and text[right_end] not in _TERMINAL_PUNCTUATION + ",，":
        right_end += 1
    left_text = text[start:index]
    right_text = text[index + 1 : right_end]
    if _STRUCTURED_FIELD_PATTERN.match(right_text):
        return True
    return bool(
        _NEGATIVE_PATTERN.search(left_text)
        and not _NEGATIVE_PATTERN.search(right_text)
        and _EXPLICIT_POSITIVE_PATTERN.search(right_text)
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _contains_fact_content(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _iter_trimmed_spans(text: str) -> Iterator[tuple[int, int]]:
    start = 0
    index = 0
    while index < len(text):
        conjunction = _CLAUSE_CONJUNCTION_PATTERN.match(text, index)
        if conjunction is not None and index > start:
            left, right = _trim_span(text, start, index)
            if left < right and _contains_fact_content(text[left:right]):
                yield left, right
            start = index
            index = conjunction.end()
            continue

        if _is_soft_clause_boundary(text, index, start):
            raw_end = index + 1
            left, right = _trim_span(text, start, raw_end)
            if left < right and _contains_fact_content(text[left:right]):
                yield left, right
            start = raw_end
            index = raw_end
            continue

        if not _is_boundary(text, index):
            index += 1
            continue

        raw_end = index + 1
        if text[index] in _TERMINAL_PUNCTUATION:
            while (
                raw_end < len(text)
                and text[raw_end] in _TERMINAL_PUNCTUATION
            ):
                raw_end += 1
        left, right = _trim_span(text, start, raw_end)
        if left < right and _contains_fact_content(text[left:right]):
            yield left, right
        start = raw_end
        index = raw_end

    left, right = _trim_span(text, start, len(text))
    if left < right and _contains_fact_content(text[left:right]):
        yield left, right


def _classify_fact(text: str) -> tuple[FactType, Polarity, Temporality]:
    polarity = Polarity.PRESENT
    if _NEGATIVE_PATTERN.search(text):
        polarity = Polarity.ABSENT
    elif _UNCERTAIN_PATTERN.search(text):
        polarity = Polarity.UNCERTAIN

    temporality = Temporality.CURRENT
    if _PAST_PATTERN.search(text):
        temporality = Temporality.PAST
    elif _FUTURE_PATTERN.search(text):
        temporality = Temporality.FUTURE

    if polarity is Polarity.ABSENT:
        fact_type = FactType.NEGATIVE_FINDING
    else:
        fact_type = FactType.OTHER
        for candidate_type, pattern in _TYPE_PATTERNS:
            if pattern.search(text):
                fact_type = candidate_type
                break
    return fact_type, polarity, temporality


def canonical_fact_signature(fact: CaseFact) -> str:
    """Return a conservative identity for exact duplicate evidence spans."""

    normalized = unicodedata.normalize("NFKC", fact.text).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    payload = "|".join(
        (
            normalized,
            fact.polarity.value,
            fact.temporality.value,
            fact.modality.value,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_clinical_subject(text: str) -> str:
    """Best-effort key for direct positive/negative fact contradictions.

    This intentionally detects only simple lexical contradictions. More
    complex clinical contradiction detection belongs to a later verifier.
    """

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _ASSERTION_WORDS_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized)
    return " ".join(normalized.split())


class FactIndex(Mapping[str, CaseFact]):
    """Immutable lookup of source-faithful, addressable case facts."""

    def __init__(
        self,
        case_id: str,
        source_text: str,
        facts: Iterable[CaseFact],
    ) -> None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise SchemaValidationError("case_id must not be empty")
        if not isinstance(source_text, str) or not source_text.strip():
            raise SchemaValidationError("case text must not be empty")
        by_id: dict[str, CaseFact] = {}
        ordered: list[CaseFact] = []
        previous_end = -1
        for fact in facts:
            if not isinstance(fact, CaseFact):
                raise SchemaValidationError(
                    "fact index entries must be CaseFact objects"
                )
            if fact.case_id != case_id:
                raise SchemaValidationError(
                    f"{fact.fact_id} belongs to case {fact.case_id!r}, "
                    f"not {case_id!r}"
                )
            if fact.fact_id in by_id:
                raise DuplicateIdentifierError(
                    f"duplicate fact_id: {fact.fact_id}"
                )
            if source_text[fact.start_char : fact.end_char] != fact.text:
                raise SchemaValidationError(
                    f"{fact.fact_id} does not exactly match its source span"
                )
            if fact.end_char > len(source_text):
                raise SchemaValidationError(
                    f"{fact.fact_id} extends beyond the case text"
                )
            if fact.start_char < previous_end:
                raise SchemaValidationError(
                    "fact spans must be ordered and non-overlapping"
                )
            by_id[fact.fact_id] = fact
            ordered.append(fact)
            previous_end = fact.end_char
        if not ordered:
            raise SchemaValidationError("fact index must contain at least one fact")

        self._case_id = case_id
        self._source_text = source_text
        self._facts = tuple(ordered)
        self._by_id = by_id
        self._sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    @classmethod
    def from_text(
        cls,
        source_text: str,
        *,
        case_id: str,
        prefix: str = "F",
    ) -> "FactIndex":
        if not isinstance(source_text, str) or not source_text.strip():
            raise SchemaValidationError("case text must not be empty")
        if not isinstance(prefix, str) or not prefix.isalpha():
            raise SchemaValidationError("fact ID prefix must contain letters only")

        spans = list(_iter_trimmed_spans(source_text))
        width = max(3, len(str(len(spans))))
        facts: list[CaseFact] = []
        for position, (start, end) in enumerate(spans, start=1):
            text = source_text[start:end]
            fact_type, polarity, temporality = _classify_fact(text)
            facts.append(
                CaseFact(
                    fact_id=f"{prefix.upper()}{position:0{width}d}",
                    case_id=case_id,
                    text=text,
                    start_char=start,
                    end_char=end,
                    fact_type=fact_type,
                    polarity=polarity,
                    temporality=temporality,
                    modality=Modality.TEXT,
                )
            )
        return cls(case_id=case_id, source_text=source_text, facts=facts)

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def source_text(self) -> str:
        return self._source_text

    @property
    def source_sha256(self) -> str:
        return self._sha256

    @property
    def facts(self) -> tuple[CaseFact, ...]:
        return self._facts

    def __getitem__(self, fact_id: str) -> CaseFact:
        try:
            return self._by_id[fact_id]
        except KeyError as exc:
            raise UnknownFactError(f"unknown fact_id: {fact_id}") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def validate_refs(self, fact_ids: Iterable[str]) -> None:
        for fact_id in fact_ids:
            if fact_id not in self._by_id:
                raise UnknownFactError(f"unknown fact_id: {fact_id}")

    def format_for_prompt(self) -> str:
        return "\n".join(
            f"{fact.fact_id} [{fact.fact_type.value}/"
            f"{fact.polarity.value}/{fact.temporality.value}]: {fact.text}"
            for fact in self._facts
        )

    def hard_conflict_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for index, left in enumerate(self._facts):
            left_key = canonical_clinical_subject(left.text)
            if not left_key:
                continue
            for right in self._facts[index + 1 :]:
                if left.temporality is not right.temporality:
                    continue
                if canonical_clinical_subject(right.text) != left_key:
                    continue
                if {left.polarity, right.polarity} == {
                    Polarity.PRESENT,
                    Polarity.ABSENT,
                }:
                    pairs.append((left.fact_id, right.fact_id))
        return tuple(pairs)
