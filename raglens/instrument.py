"""SDK auto-instrumentation (FR1.5, partial / Tier 2).

Wraps the raw OpenAI and Anthropic client calls so that, when they run *inside*
a user's generator within an active capture, token counts, model name, latency,
and (best-effort) cost are attached to the trace's generation stage — without the
user threading usage data through by hand.

This is intentionally narrow in v1: only the two most common LLM SDKs, only chat
/ messages completions. v2 broadens this to vector DBs and frameworks (FR2.5).
The SDKs are optional dependencies; if they are not installed, the corresponding
``patch_*`` call is a safe no-op.

Pricing is approximate and user-overridable; cost is a convenience, not a billing
source of truth.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .capture import active_capture

# Approximate USD per 1K tokens (prompt, completion). Override via set_pricing().
# These are coarse defaults; users should set their own for accuracy.
_PRICING: Dict[str, tuple] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-haiku": (0.00025, 0.00125),
}

_patched = {"openai": False, "anthropic": False}


def set_pricing(model: str, prompt_per_1k: float, completion_per_1k: float) -> None:
    """Register/override per-1K-token pricing for a model (prefix match)."""
    _PRICING[model] = (prompt_per_1k, completion_per_1k)


def _estimate_cost(model: Optional[str], prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    if not model:
        return None
    rates = None
    for key, val in _PRICING.items():
        if model.startswith(key):
            rates = val
            break
    if rates is None:
        return None
    p = (prompt_tokens or 0) / 1000.0 * rates[0]
    c = (completion_tokens or 0) / 1000.0 * rates[1]
    return round(p + c, 6)


def _record_usage(model, prompt_tokens, completion_tokens, latency_ms) -> None:
    cap = active_capture()
    if cap is None:
        return
    cap.record_llm_usage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
        latency_ms=latency_ms,
    )


def patch_openai() -> bool:
    """Patch ``openai`` chat completions to record usage. No-op if not installed.

    Returns True if patching happened. Safe to call multiple times (idempotent).
    """
    if _patched["openai"]:
        return True
    try:
        from openai.resources.chat import completions as _comp  # type: ignore
    except Exception:
        return False

    original = _comp.Completions.create

    def wrapped(self, *args: Any, **kwargs: Any):  # noqa: ANN001
        t0 = time.perf_counter()
        resp = original(self, *args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            model = getattr(resp, "model", None) or kwargs.get("model")
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            _record_usage(model, pt, ct, latency_ms)
        except Exception:
            # Observation must never break the host call (anticipates NFR3.4).
            pass
        return resp

    _comp.Completions.create = wrapped  # type: ignore[assignment]
    _patched["openai"] = True
    return True


def patch_anthropic() -> bool:
    """Patch ``anthropic`` messages.create to record usage. No-op if not installed."""
    if _patched["anthropic"]:
        return True
    try:
        from anthropic.resources import messages as _msg  # type: ignore
    except Exception:
        return False

    original = _msg.Messages.create

    def wrapped(self, *args: Any, **kwargs: Any):  # noqa: ANN001
        t0 = time.perf_counter()
        resp = original(self, *args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            model = getattr(resp, "model", None) or kwargs.get("model")
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "input_tokens", None) if usage else None
            ct = getattr(usage, "output_tokens", None) if usage else None
            _record_usage(model, pt, ct, latency_ms)
        except Exception:
            pass
        return resp

    _msg.Messages.create = wrapped  # type: ignore[assignment]
    _patched["anthropic"] = True
    return True


def auto_instrument() -> Dict[str, bool]:
    """Best-effort: patch whichever supported SDKs are importable.

    Returns a map of SDK name → whether it was patched.
    """
    return {"openai": patch_openai(), "anthropic": patch_anthropic()}
