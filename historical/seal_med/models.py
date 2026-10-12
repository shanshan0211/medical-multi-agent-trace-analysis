"""Injectable model boundary for SEAL-Med orchestration.

This module contains no network client. Production callers inject the existing
``llm_client.chat_completion`` function through ``ChatCompletionModelClient``.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .exceptions import SchemaValidationError
from .orchestration import CallAudit, Stage

_JSON_FENCE_RE = None  # populated lazily below


def _salvage_json_content(content: str) -> str:
    """Salvage a JSON object from GLM prose when json_object is ignored.

    Project convention (see llm_client / delphi parsers): GLM-5.x frequently
    ignores response_format=json_object and answers with leading prose. If the
    content is not already valid JSON, extract the first balanced {...} block;
    if none exists, leave the content untouched so strict validation raises.
    """
    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return content
    import re

    start = stripped.find("{")
    if start == -1:
        return content
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    return content


@dataclass(frozen=True)
class ModelReply:
    content: str
    model_id: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelClient(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_family(self) -> str: ...

    def complete(
        self,
        messages: tuple[Mapping[str, str], ...],
        *,
        temperature: float = 0.0,
    ) -> ModelReply: ...


CompletionCallable = Callable[..., Mapping[str, Any]]


class ChatCompletionModelClient:
    """Adapter around an OpenAI-compatible chat completion callable."""

    def __init__(
        self,
        completion: CompletionCallable,
        *,
        model_id: str,
        model_family: str | None = None,
        temperature: float = 0.0,
        completion_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(completion):
            raise SchemaValidationError("completion must be callable")
        if not isinstance(model_id, str) or not model_id.strip():
            raise SchemaValidationError("model_id must not be blank")
        if model_family is None or not isinstance(model_family, str) or not model_family.strip():
            raise SchemaValidationError(
                "model_family must be explicitly declared for correlation checks"
            )
        self._completion = completion
        self._model_id = model_id.strip()
        self._model_family = model_family.strip()
        self.temperature = float(temperature)
        self.completion_kwargs = dict(completion_kwargs or {})
        self._deadline_monotonic: float | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_family(self) -> str:
        return self._model_family

    def set_deadline(self, deadline_monotonic: float | None) -> None:
        if deadline_monotonic is not None and not isinstance(deadline_monotonic, (int, float)):
            raise SchemaValidationError("model deadline must be a monotonic timestamp or None")
        self._deadline_monotonic = (
            float(deadline_monotonic) if deadline_monotonic is not None else None
        )

    def complete(
        self,
        messages: tuple[Mapping[str, str], ...],
        *,
        temperature: float = 0.0,
        overrides: Mapping[str, Any] | None = None,
    ) -> ModelReply:
        completion_kwargs = dict(self.completion_kwargs)
        # Per-call overrides (e.g. an escalated rescue max_tokens on schema
        # retries) take precedence over the client-level defaults without
        # mutating the shared client instance.
        if overrides:
            completion_kwargs.update(dict(overrides))
        if self._deadline_monotonic is not None:
            remaining = self._deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise SchemaValidationError("case wall-clock deadline exhausted before model call")
            configured_timeout = completion_kwargs.get("timeout")
            if (
                isinstance(configured_timeout, (int, float))
                and not isinstance(configured_timeout, bool)
                and configured_timeout > 0
            ):
                completion_kwargs["timeout"] = min(float(configured_timeout), remaining)
            else:
                completion_kwargs["timeout"] = remaining
        started = time.perf_counter()
        call_kwargs = {
            "messages": [dict(message) for message in messages],
            "model": self.model_id,
            "temperature": temperature,
            **completion_kwargs,
        }
        if self._deadline_monotonic is None:
            response = self._completion(**call_kwargs)
        else:
            response = self._complete_before_deadline(
                call_kwargs,
                deadline_monotonic=self._deadline_monotonic,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SchemaValidationError("model response has no choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason")
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            reasoning_len = len(reasoning) if isinstance(reasoning, str) else 0
            usage = response.get("usage") or {}
            raise SchemaValidationError(
                "model response content must not be blank"
                f" (finish_reason={finish_reason!r}, reasoning_len={reasoning_len},"
                f" usage={usage})"
            )
        usage = response.get("usage", {})
        content = _salvage_json_content(content)
        return ModelReply(
            content=content,
            model_id=self.model_id,
            latency_ms=latency_ms,
            input_tokens=_optional_token(usage, "prompt_tokens"),
            output_tokens=_optional_token(usage, "completion_tokens"),
        )

    def _complete_before_deadline(
        self,
        call_kwargs: Mapping[str, Any],
        *,
        deadline_monotonic: float,
    ) -> Mapping[str, Any]:
        """Enforce a hard outer deadline around clients with idle-only timeouts."""

        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                result_queue.put((True, self._completion(**dict(call_kwargs))))
            except BaseException as exc:
                result_queue.put((False, exc))

        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise SchemaValidationError("case wall-clock deadline exhausted before model call")
        worker = threading.Thread(
            target=invoke,
            name="seal-med-model-call",
            daemon=True,
        )
        worker.start()
        try:
            succeeded, value = result_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise SchemaValidationError(
                "case wall-clock deadline exhausted during model call"
            ) from exc
        if not succeeded:
            raise value
        if not isinstance(value, Mapping):
            raise SchemaValidationError("model completion must return a mapping")
        return value


class StaticModelClient:
    """Deterministic in-memory client for smoke tests and reproducible demos."""

    def __init__(self, replies: list[str], *, model_id: str = "static-model", model_family: str = "static") -> None:
        if not replies:
            raise SchemaValidationError("StaticModelClient requires at least one reply")
        self._replies = list(replies)
        self._model_id = model_id
        self._model_family = model_family
        self._position = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_family(self) -> str:
        return self._model_family

    def complete(self, messages: tuple[Mapping[str, str], ...], *, temperature: float = 0.0) -> ModelReply:
        if self._position >= len(self._replies):
            raise RuntimeError("StaticModelClient reply budget exhausted")
        content = self._replies[self._position]
        self._position += 1
        return ModelReply(content=content, model_id=self.model_id, latency_ms=0)


def make_call_audit(
    *,
    call_id: str,
    stage: Stage,
    messages: tuple[Mapping[str, str], ...],
    reply: ModelReply,
    schema_errors: tuple[str, ...] = (),
) -> CallAudit:
    prompt_bytes = json.dumps(
        [dict(message) for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return CallAudit(
        call_id=call_id,
        stage=stage,
        model_id=reply.model_id,
        prompt_hash=hashlib.sha256(prompt_bytes).hexdigest(),
        response_hash=hashlib.sha256(reply.content.encode("utf-8")).hexdigest(),
        latency_ms=reply.latency_ms,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        schema_errors=schema_errors,
    )


TOKEN_INNER_JSON_KEYS = ("primary_diagnosis", "top3_differentials")


def _extract_json_objects(text: str) -> list[str]:
    """Return every complete, valid top-level JSON object embedded in *text*."""
    found: list[str] = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:j + 1]
                    try:
                        json.loads(candidate)
                        found.append(candidate)
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            i = start + 1
    return found


def _prefer_answer_json(objects: Sequence[str]) -> str | None:
    """Among embedded JSON objects, return the last one shaped like a turn answer."""
    shaped = [o for o in objects if all(k in o for k in TOKEN_INNER_JSON_KEYS)]
    pool = shaped or list(objects)
    return pool[-1] if pool else None


def parse_json_object(content: str) -> Mapping[str, Any]:
    """Parse one JSON object.

    Strict parsing is primary: fenced or fallback-filled output is otherwise
    rejected.  As a recovery for prose-emitting peers (glm-family can pad a
    valid answer object with chain-of-thought), when strict parsing fails we
    fall back to the last embedded object that is shaped like a turn answer.
    """

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        embedded = _extract_json_objects(content)
        chosen = _prefer_answer_json(embedded) if embedded else None
        if chosen is None:
            raise SchemaValidationError(
                f"model output is not strict JSON: {exc.msg}"
            ) from exc
        payload = json.loads(chosen)
    if not isinstance(payload, Mapping):
        raise SchemaValidationError("model output must be one JSON object")
    return payload


def _optional_token(usage: object, key: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(key)
    return value if type(value) is int and value >= 0 else None
