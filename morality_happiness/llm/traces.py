from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass
class LLMTraceEntry:
    timestamp: str
    provider: str
    model: str
    stage: str
    prompt: str
    response: str
    ok: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TRACE_LOCK = Lock()
_TRACE_BUFFER: deque[LLMTraceEntry] = deque(maxlen=200)


def append_llm_trace(
    *,
    provider: str,
    model: str,
    stage: str,
    prompt: str,
    response: str,
    ok: bool,
    error: str = "",
) -> None:
    entry = LLMTraceEntry(
        timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        provider=provider,
        model=model,
        stage=stage,
        prompt=str(prompt),
        response=str(response),
        ok=bool(ok),
        error=str(error),
    )
    with _TRACE_LOCK:
        _TRACE_BUFFER.append(entry)


def list_llm_traces(limit: int = 80) -> list[dict[str, Any]]:
    with _TRACE_LOCK:
        items = list(_TRACE_BUFFER)
    return [item.to_dict() for item in items[-max(1, min(int(limit), 200)) :]]


def clear_llm_traces() -> None:
    with _TRACE_LOCK:
        _TRACE_BUFFER.clear()
