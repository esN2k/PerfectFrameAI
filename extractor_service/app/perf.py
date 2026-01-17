"""
Performance utilities for timing and GPU telemetry.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_WARNED_NO_NVIDIA_SMI = False
_PERF_CONTEXT: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "perf_context", default={}
)
_PERF_DATA: dict[str, dict] = {"totals": {}, "videos": {}, "records": []}


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in ("0", "false", "no", "off")


def perf_enabled() -> bool:
    return _env_flag("PERFECTFRAMEAI_LOG_PERF", "1")


def gpu_log_enabled() -> bool:
    return _env_flag("PERFECTFRAMEAI_LOG_GPU", "1")


def gpu_log_batch_enabled() -> bool:
    return _env_flag("PERFECTFRAMEAI_LOG_GPU_BATCH", "0")


@contextmanager
def perf_context(**kwargs: str):
    """Attach contextual metadata (e.g. video) to performance logs."""
    current = _PERF_CONTEXT.get()
    merged = {**current, **kwargs}
    token = _PERF_CONTEXT.set(merged)
    try:
        yield
    finally:
        _PERF_CONTEXT.reset(token)


def _parse_details(details: str | None) -> dict[str, str]:
    if not details:
        return {}
    tokens = details.replace(",", " ").split()
    parsed: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and value:
            parsed[key.strip()] = value.strip()
    return parsed


def _record_perf(label: str, duration: float, details: str | None) -> None:
    context = _PERF_CONTEXT.get()
    parsed = _parse_details(details)
    video = context.get("video") or parsed.get("video")
    _PERF_DATA["totals"][label] = _PERF_DATA["totals"].get(label, 0.0) + duration
    if video:
        video_data = _PERF_DATA["videos"].setdefault(video, {})
        video_data[label] = video_data.get(label, 0.0) + duration
    _PERF_DATA["records"].append(
        {
            "label": label,
            "duration": duration,
            "details": details,
            "context": context,
            "timestamp": time.time(),
        }
    )
    perf_path = os.getenv("PERFECTFRAMEAI_PERF_PATH")
    if perf_path:
        _write_perf(Path(perf_path))


def _write_perf(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "totals": _PERF_DATA["totals"],
            "videos": _PERF_DATA["videos"],
            "records": _PERF_DATA["records"],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to write perf data to %s: %s", path, exc)


def log_perf_metric(label: str, duration: float, details: str | None = None) -> None:
    if not perf_enabled():
        return
    suffix = f" ({details})" if details else ""
    logger.info("PERF %s: %.3fs%s", label, duration, suffix)
    _record_perf(label, duration, details)


def profile_time(
    label: str | None = None,
    details_fn: Callable[..., str | None] | None = None,
):
    """Decorator that records performance timing for functions and generators."""
    def decorator(func):
        def _details(*args, **kwargs) -> str | None:
            if details_fn is None:
                return None
            try:
                return details_fn(*args, **kwargs)
            except Exception:  # pragma: no cover - defensive path
                return None

        def _wrap_generator(gen, start, details):
            try:
                for item in gen:
                    yield item
            finally:
                duration = time.perf_counter() - start
                log_perf_metric(label or func.__name__, duration, details)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not perf_enabled():
                return func(*args, **kwargs)
            details = _details(*args, **kwargs)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            if hasattr(result, "__iter__") and hasattr(result, "__next__"):
                return _wrap_generator(result, start, details)
            duration = time.perf_counter() - start
            log_perf_metric(label or func.__name__, duration, details)
            return result

        return wrapper

    return decorator


@contextmanager
def perf_timer(label: str, details: str | None = None):
    if not perf_enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        log_perf_metric(label, duration, details)


def log_gpu_stats(phase: str, details: str | None = None) -> None:
    if not gpu_log_enabled():
        return
    global _WARNED_NO_NVIDIA_SMI
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        if not _WARNED_NO_NVIDIA_SMI:
            logger.warning("nvidia-smi not available for GPU telemetry: %s", exc)
            _WARNED_NO_NVIDIA_SMI = True
        return
    payload = result.stdout.strip()
    if not payload:
        return
    suffix = f" ({details})" if details else ""
    logger.info("GPU %s: %s%s", phase, payload.replace("\n", " | "), suffix)
