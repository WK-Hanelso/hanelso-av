from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict


def _env_enabled() -> bool:
    value = os.getenv("SIM_PROFILE", "")
    return value.lower() not in {"", "0", "false", "no", "off"}


class _PerfCollector:
    def __init__(self) -> None:
        self.enabled = _env_enabled()
        self._stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total_ns": 0, "count": 0}
        )

    def reset(self) -> None:
        self.enabled = _env_enabled()
        self._stats.clear()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            stat = self._stats[name]
            stat["total_ns"] += elapsed
            stat["count"] += 1

    def snapshot(self) -> Dict[str, Dict[str, float | int]]:
        if not self.enabled:
            return {}
        snapshot: Dict[str, Dict[str, float | int]] = {}
        for name, stat in sorted(self._stats.items()):
            total_ns = int(stat["total_ns"])
            count = int(stat["count"])
            snapshot[name] = {
                "count": count,
                "total_ms": round(total_ns / 1e6, 3),
                "avg_ms": round((total_ns / count) / 1e6, 3) if count else 0.0,
            }
        return snapshot


_COLLECTOR = _PerfCollector()


def reset() -> None:
    _COLLECTOR.reset()


def snapshot() -> Dict[str, Dict[str, float | int]]:
    return _COLLECTOR.snapshot()


@contextmanager
def section(name: str):
    with _COLLECTOR.section(name):
        yield
