"""High-level reconstruction runner with streaming Parquet output."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .counters import (
    merge_counters,
    print_counters,
    reset_counters,
    snapshot_counters,
)
from .io import event_stream, n_chunk_events


class ParquetSink:
    """Stream DataFrames into a single Parquet file with bounded memory.

    Buffers ~buffer_bytes of rows before flushing a row group, so huge runs
    neither hold all output in memory nor produce thousands of tiny row
    groups. Always produces a file on close (empty if no rows arrived).
    """

    def __init__(self, path: str | Path, buffer_bytes: int = 128 * 1024**2):
        self.path = Path(path)
        self.buffer_bytes = buffer_bytes
        self._writer = None
        self._buffer: list[pd.DataFrame] = []
        self._buffered_bytes = 0
        self.rows_written = 0

    def write(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return
        self._buffer.append(df)
        self._buffered_bytes += int(df.memory_usage(deep=False).sum())
        if self._buffered_bytes >= self.buffer_bytes:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(
            pd.concat(self._buffer, ignore_index=True), preserve_index=False
        )
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        self.rows_written += len(table)
        self._buffer = []
        self._buffered_bytes = 0

    def close(self):
        self._flush()
        if self._writer is not None:
            self._writer.close()
        else:
            # no rows: still produce a valid (empty) parquet file
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(self.path, index=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _worker_init():
    """Fresh state per worker: no inherited tree handles or counters."""
    from .io import _open_tree

    _open_tree.cache_clear()
    reset_counters()


def _process_chunk(fn: Callable, chunk: dict):
    """Worker side: run one chunk, return (result, counters delta)."""
    reset_counters()
    result = fn(chunk)
    return result, snapshot_counters()


def run_reconstruction(
    reconstruction_function: Callable,
    input_data: str | Path = "",
    out: str | Path | None = None,
    max_events: int | None = None,
    chunk_size: int = 100,
    workers: int = 1,
    print_throughput: bool = False,
) -> list[Any] | Path | None:
    """Run a reconstruction function over chunk cursors.

    The function receives a chunk cursor and loads what it needs via the
    component loaders. With out=None, non-None results are collected and
    returned as a list. With out=<path>, DataFrame results are streamed
    into a single Parquet file (constant memory) and the path is returned.

    With workers > 1, chunks are processed by a process pool; results are
    merged in completion order (row order is not reproducible across runs,
    physics content is). The reconstruction function must be a module-level
    (picklable) function.
    """
    t0 = time.perf_counter()
    n_events = 0
    results: list[Any] = []
    sink = ParquetSink(out) if out is not None else None

    def _consume(ret):
        if ret is None:
            return
        if sink is not None:
            sink.write(ret)
        else:
            results.append(ret)

    try:
        if workers <= 1:
            for chunk in event_stream(
                input_data, chunk_size=chunk_size, max_events=max_events
            ):
                _consume(reconstruction_function(chunk))
                n_events += n_chunk_events(chunk)
        else:
            chunks = list(
                event_stream(
                    input_data, chunk_size=chunk_size, max_events=max_events
                )
            )
            n_events = sum(n_chunk_events(c) for c in chunks)
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_worker_init
            ) as pool:
                futures = [
                    pool.submit(_process_chunk, reconstruction_function, c)
                    for c in chunks
                ]
                for fut in as_completed(futures):
                    ret, snapshot = fut.result()
                    merge_counters(snapshot)
                    _consume(ret)
    finally:
        if sink is not None:
            sink.close()

    if print_throughput:
        elapsed = time.perf_counter() - t0
        rate = n_events / elapsed if elapsed > 0 else float("inf")
        print(
            f"Processed {n_events} events in {elapsed:.2f}s ({rate:.1f} evt/s)"
        )
    print_counters()

    if sink is not None:
        return Path(out)
    return results if results else None
