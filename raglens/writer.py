"""Versioned JSONL (de)serialization for RagTrace (FR1.3).

One JSON object per line. JSONL is chosen deliberately: it is append-friendly
(good for streaming capture), diff-friendly, and trivially consumable by other
tools — honoring the "treat the trace as a software artifact" framing.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, List, Union

from .ir import RagTrace

PathLike = Union[str, "os.PathLike[str]"]


def write_traces(path: PathLike, traces: Iterable[RagTrace], append: bool = False) -> None:
    """Write traces to a JSONL file (one trace per line)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def append_trace(path: PathLike, trace: RagTrace) -> None:
    """Append a single trace to a JSONL file."""
    write_traces(path, [trace], append=True)


def iter_traces(path: PathLike) -> Iterator[RagTrace]:
    """Lazily yield traces from a JSONL file (streaming; v2-friendly)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield RagTrace.from_dict(json.loads(line))


def read_traces(path: PathLike) -> List[RagTrace]:
    """Read all traces from a JSONL file into a list."""
    return list(iter_traces(path))
