"""CLI: python -m pid_performance --input LABEL file.parquet ..."""

from __future__ import annotations

import argparse
from typing import Sequence

from .comparison import PIDComparison
from .performance import PIDPerformance


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PID performance / comparison from parquet files"
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("LABEL", "PARQUET"),
        help="Repeatable: --input Run3 a.parquet --input Upgrade b.parquet",
    )
    parser.add_argument("--out-dir", default="public/pid_performance")
    parser.add_argument("--out-tag", default="pid_performance")
    parser.add_argument("--signal", default="K+")
    parser.add_argument("--background", default="pi+")
    parser.add_argument("--dll-field", default="rich_dll_kaon")
    parser.add_argument("--target", type=float, action="append", default=None)
    args = parser.parse_args(argv)

    common = dict(
        out_dir=args.out_dir,
        signal_id=args.signal,
        background_id=args.background,
        dll_field=args.dll_field,
    )
    if not args.input:
        raise SystemExit("Pass at least one --input LABEL PARQUET")

    if len(args.input) == 1:
        label, path = args.input[0]
        performance = PIDPerformance.from_parquet(
            path, out_tag=args.out_tag, label=label, **common
        )
        performance.run_all(targets=args.target)
    else:
        files = {label: path for label, path in args.input}
        PIDComparison.from_parquets(
            files, out_tag=args.out_tag, **common
        ).run_all(targets=args.target)


if __name__ == "__main__":
    main()
