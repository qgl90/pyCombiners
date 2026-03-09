"""High-level reconstruction runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .counters import print_counters
from .io import load_events_in_slices


def run_reconstruction(
    reconstruction_function: Callable,
    input_data: str | Path = "",
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
    slice_size: int = 100,
    print_throughput: bool = False,
) -> list | None:
    """Run a reconstruction function over event slices."""
    t0 = time.perf_counter()
    n_events = 0
    results = []
    for tracks, pvs, event_info in load_events_in_slices(
        input_data,
        tree_name,
        max_events=max_events,
        slice_size=slice_size,
    ):
        events = {"tracks": tracks, "pvs": pvs, **event_info}
        n_events += len(event_info["run_number"])
        ret = reconstruction_function(events)
        if ret is not None:
            results.append(ret)
    if print_throughput:
        elapsed = time.perf_counter() - t0
        print(
            f"Processed {n_events} events in {elapsed:.2f}s ({n_events / elapsed:.1f} evt/s)"
        )
    print_counters()
    return results if results else None
