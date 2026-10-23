"""Token/cost usage accounting shared by all providers.

Providers call `record(...)` after each successful request. Totals are kept per (provider, model) in-process;
if a JSONL path is set (`set_log_path` or env LLM_PROVIDERS_USAGE_LOG), every call is appended as one line.
Cost is the provider-reported cost (OpenRouter `usage.cost`) when available, else None (no price tables here).
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

_lock = threading.Lock()
_totals: Dict[tuple, Dict[str, float]] = {}
_log_path: Optional[str] = os.environ.get("LLM_PROVIDERS_USAGE_LOG") or None


def set_log_path(path: Optional[str]) -> None:
    """Append every recorded call to this JSONL file (None disables)."""
    global _log_path
    _log_path = path


def record(provider: str, model: str, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
           cost: Optional[float] = None, latency_s: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Record one call's usage."""
    entry = {"ts": time.time(), "provider": provider, "model": model, "input_tokens": input_tokens,
             "output_tokens": output_tokens, "cost": cost, "latency_s": latency_s}
    if extra:
        entry.update(extra)
    with _lock:
        t = _totals.setdefault((provider, model), {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                                                   "latency_s": 0.0})
        t["calls"] += 1
        t["input_tokens"] += input_tokens or 0
        t["output_tokens"] += output_tokens or 0
        t["cost"] += cost or 0.0
        t["latency_s"] += latency_s or 0.0
        if _log_path:
            os.makedirs(os.path.dirname(os.path.abspath(_log_path)), exist_ok=True)
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def totals() -> Dict[str, Dict[str, float]]:
    """Accumulated usage per "provider:model"."""
    with _lock:
        return {f"{p}:{m}": dict(v) for (p, m), v in _totals.items()}


def reset() -> None:
    """Clear the in-process totals."""
    with _lock:
        _totals.clear()


def _safe(fn, *args, **kwargs) -> None:
    """Never let usage accounting break a provider call."""
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"llm_providers.usage: failed to record usage: {e}")
