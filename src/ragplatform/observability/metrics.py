"""Tiny dependency-free Prometheus metrics (counters + histograms) and request-id logging."""
from __future__ import annotations

import logging
import threading
from collections import defaultdict

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
log = logging.getLogger("ragplatform")


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self.hist: dict[tuple[str, tuple], list[float]] = defaultdict(lambda: [0.0] * (len(BUCKETS) + 2))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self.lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, seconds: float, **labels: str) -> None:
        with self.lock:
            h = self.hist[(name, tuple(sorted(labels.items())))]
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    h[i] += 1
            h[-2] += 1  # +Inf / count
            h[-1] += seconds  # sum

    @staticmethod
    def _lbl(labels: tuple, extra: str = "") -> str:
        parts = [f'{k}="{v}"' for k, v in labels] + ([extra] if extra else [])
        return "{" + ",".join(parts) + "}" if parts else ""

    def render(self) -> str:
        out = []
        with self.lock:
            for (name, labels), v in sorted(self.counters.items()):
                out.append(f"{name}{self._lbl(labels)} {v}")
            for (name, labels), h in sorted(self.hist.items()):
                for i, b in enumerate(BUCKETS):
                    le = f'le="{b}"'
                    out.append(f"{name}_bucket{self._lbl(labels, le)} {h[i]}")
                inf = 'le="+Inf"'
                out.append(f"{name}_bucket{self._lbl(labels, inf)} {h[-2]}")
                out.append(f"{name}_count{self._lbl(labels)} {h[-2]}")
                out.append(f"{name}_sum{self._lbl(labels)} {h[-1]}")
        return "\n".join(out) + "\n"
