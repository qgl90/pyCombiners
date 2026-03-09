"""Global counters for reconstruction diagnostics."""

from __future__ import annotations

_counter_values: dict[str, int] = {}
_rate_values: dict[str, list[int]] = {}


class _Counter:
    def __init__(self, name: str):
        self._name = name

    def add(self, n: int = 1):
        _counter_values[self._name] = _counter_values.get(self._name, 0) + n


class _RateCounter:
    def __init__(self, name: str):
        self._name = name

    def add(self, n_pass: int, n_total: int):
        if self._name not in _rate_values:
            _rate_values[self._name] = [0, 0]
        _rate_values[self._name][0] += n_pass
        _rate_values[self._name][1] += n_total


def counters(name: str) -> _Counter:
    return _Counter(name)


def rate_counters(name: str) -> _RateCounter:
    return _RateCounter(name)


def print_counters():
    for name, n in _counter_values.items():
        print(f"{name}: {n}")
    for name, (n_pass, n_total) in _rate_values.items():
        pct = n_pass / n_total * 100 if n_total > 0 else 0.0
        print(f"{name}: {n_pass}/{n_total} ({pct:.1f}%)")


def reset_counters():
    _counter_values.clear()
    _rate_values.clear()
