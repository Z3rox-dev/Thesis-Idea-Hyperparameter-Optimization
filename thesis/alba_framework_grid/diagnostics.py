"""Diagnostics helpers for ALBA tracing.

This module is intentionally lightweight and dependency-free (numpy optional).
It provides utilities to:
- convert trace payloads (with numpy objects) into JSON-serializable structures
- write trace events to a JSONL file (one event per line)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


JsonLike = Union[None, bool, int, float, str, Dict[str, Any], list]


def to_jsonable(obj: Any) -> JsonLike:
    """Best-effort conversion of (possibly numpy-heavy) objects into JSON-serializable values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if np is not None:
        try:
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = to_jsonable(v)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    return str(obj)


class TraceJSONLWriter:
    """Callable trace hook that appends JSONL events to a file."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        flush: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)
        self._flush = bool(flush)

    def __call__(self, trace: Dict[str, Any]) -> None:
        self.write(trace)

    def write(self, trace: Dict[str, Any]) -> None:
        json.dump(to_jsonable(trace), self._fh)
        self._fh.write("\n")
        if self._flush:
            try:
                self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "TraceJSONLWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


def make_jsonl_trace_hooks(
    path: Union[str, Path],
    *,
    flush: bool = True,
) -> tuple[Callable[[Dict[str, Any]], None], Callable[[Dict[str, Any]], None], TraceJSONLWriter]:
    """Convenience: returns (trace_hook, trace_hook_tell, writer) that share the same JSONL sink."""
    writer = TraceJSONLWriter(path, flush=flush)
    return writer, writer, writer

