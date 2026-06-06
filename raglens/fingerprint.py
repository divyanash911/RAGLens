"""config_fingerprint — reproducibility & drift-detection primitive (FR1.4).

A ``config_fingerprint`` is a short, deterministic hash of the parts of a RAG
pipeline's configuration that affect its outputs: corpus version, embedding
model, chunking parameters, retrieval settings, and the prompt template.

Two traces with the same fingerprint were produced under the same configuration,
so differences in their behaviour are attributable to inputs (or LLM
nondeterminism), not to config changes. v2 drift detection compares behaviour
*across* fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

# The canonical fields that define a pipeline's identity. Unknown extra keys are
# still included (sorted) so users can extend without losing determinism.
CANONICAL_FIELDS = (
    "corpus_version",
    "embedding_model",
    "chunking",
    "retrieval",
    "reranker",
    "prompt_template",
    "generation_model",
)


def _canonicalize(config: Dict[str, Any]) -> str:
    """Serialize a config dict to a stable, canonical JSON string.

    Sorting keys and using a fixed separator means logically-equal configs always
    produce the same bytes regardless of insertion order or whitespace.
    """
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(config: Dict[str, Any], length: int = 12) -> str:
    """Return a short hex fingerprint for a pipeline config.

    Args:
        config: pipeline configuration. Any JSON-serializable dict is accepted;
            ``CANONICAL_FIELDS`` are merely a documented convention.
        length: number of hex characters to keep (default 12).
    """
    if not config:
        config = {}
    canonical = _canonicalize(config)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]
