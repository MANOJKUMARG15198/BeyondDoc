"""Request tracing (ADR-008): Langfuse when MEDRAG_TRACING=langfuse, otherwise a no-op. Never breaks a request.

Privacy (CLAUDE.md rule 9): traces carry metadata only — stage timings, model, token usage, chunk/document ids,
plan reasons, coverage, abstention and text lengths. Question, answer and passage text are added only when
MEDRAG_TRACE_CONTENT=true (synthetic demo inputs), cut to TEXT_LIMIT characters, and every value passes a mask that
removes anything shaped like a credential before it leaves the process.

Design: one trace per request. The root observation is opened once and later stages attach to it by trace id
(`trace_context`), each inside its own short `with` block. That keeps OpenTelemetry context changes inside one
synchronous step, which matters for the chat turn: it is a generator streamed by FastAPI, and a context opened in
one step and closed in another would be detached in a different thread.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import Any

log = logging.getLogger("medical_rag.tracing")

TEXT_LIMIT = 2000
_SECRET = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{16,}|fw_[A-Za-z0-9]{12,}|pk-lf-[A-Za-z0-9\-]{8,}|sk-lf-[A-Za-z0-9\-]{8,}"
    r"|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|(?i:bearer\s+[A-Za-z0-9._\-]{12,})"
    r"|(?i:(api[_-]?key|password|secret|token)\s*[:=]\s*\S{6,})"
)


def mask_secrets(*, data: Any, **_: Any) -> Any:
    """Langfuse MaskFunction: replace credential-shaped substrings everywhere in the payload."""
    if isinstance(data, str):
        return _SECRET.sub("[REDACTED]", data)
    if isinstance(data, Mapping):
        return {k: mask_secrets(data=v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [mask_secrets(data=v) for v in data]
    return data


def hash_user(user_id: str | None) -> str | None:
    """Stable pseudonym for the anonymous cookie id, so the raw id never leaves the process."""
    return hashlib.sha256(f"medrag:{user_id}".encode()).hexdigest()[:16] if user_id else None


class _NoopObservation:
    trace_id: str | None = None
    id: str | None = None

    def update(self, **_: Any) -> None:
        return None


_NOOP = _NoopObservation()


class TraceHandle:
    """One request's trace. `span()` adds a child stage; `end()` closes the root."""

    def __init__(self, tracer: Tracer, root: Any = _NOOP, attrs: Mapping[str, Any] | None = None) -> None:
        self._tracer, self._root, self._attrs = tracer, root, dict(attrs or {})

    @property
    def trace_id(self) -> str | None:
        return getattr(self._root, "trace_id", None)

    @property
    def content(self) -> bool:
        return self._tracer.content

    def text(self, value: str | None) -> str | None:
        return self._tracer.text(value)

    def update(self, **fields: Any) -> None:
        self._tracer._safe(lambda: self._root.update(**_clean(fields)))

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        if self._root is _NOOP:
            yield _NOOP
            return
        with self._tracer._child(self, name, as_type, fields) as obs:
            yield obs

    def end(self, **fields: Any) -> None:
        if self._root is _NOOP:
            return
        if fields:
            self.update(**fields)
        self._tracer._safe(self._root.end)


class Tracer:
    """No-op tracer; the base for the Langfuse one. Safe to call everywhere."""

    enabled = False

    def __init__(self, content: bool = False) -> None:
        self.content = content

    def text(self, value: str | None) -> str | None:
        """Text for a trace only when content capture is on; otherwise None (metadata keeps the length)."""
        if not self.content or value is None:
            return None
        return value[:TEXT_LIMIT]

    def start(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        input: Any = None,
    ) -> TraceHandle:
        return TraceHandle(self)

    def flush(self) -> None:
        return None

    @contextmanager
    def _child(self, handle: TraceHandle, name: str, as_type: str, fields: Mapping[str, Any]) -> Iterator[Any]:
        yield _NOOP

    def _safe(self, call: Any) -> Any:
        try:
            return call()
        except Exception:  # noqa: BLE001 - tracing must never break a request
            log.warning("tracing_failed", exc_info=True)
            return None


def _clean(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self, client: Any, *, content: bool = False, environment: str | None = None) -> None:
        super().__init__(content)
        self.client, self.environment = client, environment

    @classmethod
    def from_settings(cls, settings: Any) -> Tracer:
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            log.warning("tracing_disabled", extra={"reason": "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"})
            return Tracer()
        from langfuse import Langfuse
        from langfuse.span_filter import is_langfuse_span

        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host or None,
            environment=settings.env,
            sample_rate=settings.trace_sample_rate,
            mask=mask_secrets,
            # Export only spans made through this adapter. Libraries with their own OpenTelemetry instrumentation
            # (the Anthropic SDK records one span per API call) would otherwise add duplicate spans that bypass
            # the metadata-only rule and the content switch.
            should_export_span=is_langfuse_span,
        )
        return cls(client, content=settings.trace_content, environment=settings.env)

    def _propagate(self, attrs: Mapping[str, Any]) -> Any:
        from langfuse import propagate_attributes

        return propagate_attributes(
            user_id=attrs.get("user_id"), session_id=attrs.get("session_id"), tags=list(attrs.get("tags") or [])
        )

    def start(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        input: Any = None,
    ) -> TraceHandle:
        attrs = {"user_id": hash_user(user_id), "session_id": session_id, "tags": list(tags)}

        def open_root() -> Any:
            with ExitStack() as stack:
                root = stack.enter_context(
                    self.client.start_as_current_observation(
                        name=name, as_type="span", input=input, metadata=dict(metadata or {}), end_on_exit=False
                    )
                )
                stack.enter_context(self._propagate(attrs))
                return root

        root = self._safe(open_root)
        return TraceHandle(self, root if root is not None else _NOOP, attrs)

    @contextmanager
    def _child(self, handle: TraceHandle, name: str, as_type: str, fields: Mapping[str, Any]) -> Iterator[Any]:
        stack = ExitStack()

        def open_child() -> Any:
            obs = stack.enter_context(
                self.client.start_as_current_observation(
                    trace_context={"trace_id": handle._root.trace_id, "parent_span_id": handle._root.id},
                    name=name,
                    as_type=as_type,
                    **_clean(fields),
                )
            )
            stack.enter_context(self._propagate(handle._attrs))
            return obs

        obs = self._safe(open_child)
        wrapped = _SafeObservation(self, obs) if obs is not None else _NOOP
        try:
            yield wrapped
        except BaseException as exc:
            err = exc
            self._safe(lambda: stack.__exit__(type(err), err, err.__traceback__))
            raise
        else:
            self._safe(lambda: stack.__exit__(None, None, None))

    def flush(self) -> None:
        self._safe(self.client.flush)


class _SafeObservation:
    def __init__(self, tracer: Tracer, obs: Any) -> None:
        self._tracer, self._obs = tracer, obs
        self.trace_id, self.id = getattr(obs, "trace_id", None), getattr(obs, "id", None)

    def update(self, **fields: Any) -> None:
        self._tracer._safe(lambda: self._obs.update(**_clean(fields)))


@lru_cache(maxsize=1)
def _tracer_for(key: tuple) -> Tracer:
    from medical_rag.core.config import get_settings

    settings = get_settings()
    if settings.tracing == "langfuse":
        return LangfuseTracer.from_settings(settings)
    return Tracer()


def get_tracer() -> Tracer:
    """Process-wide tracer from the settings (no-op unless MEDRAG_TRACING=langfuse and keys are set)."""
    from medical_rag.core.config import get_settings

    s = get_settings()
    return _tracer_for((s.tracing, s.langfuse_host, s.trace_content, s.trace_sample_rate))


def usage_details(meta: Mapping[str, Any]) -> dict[str, int] | None:
    """Token usage from an LLM call's metadata, in Langfuse's usage_details shape."""
    usage = {"input": meta.get("input_tokens"), "output": meta.get("output_tokens")}
    usage = {k: int(v) for k, v in usage.items() if isinstance(v, int)}
    return usage or None
