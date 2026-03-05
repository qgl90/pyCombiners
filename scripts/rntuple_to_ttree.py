#!/usr/bin/env python3
"""Convert RNTuple ROOT files to TTree format (flat+sizes branches)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import awkward as ak
import uproot


_COMPRESSION = {"zstd": uproot.ZSTD, "zlib": uproot.ZLIB, "lz4": uproot.LZ4}


def convert(
    input_path: str,
    output_path: str,
    tree_name: str = "BestLongTracks/TrackTuple",
    chunk_size: int = 100,
    compression: str = "zstd",
    level: int = 1,
) -> None:
    tree = uproot.open(f"{input_path}:{tree_name}")
    all_keys = list(tree.keys())

    comp = _COMPRESSION[compression](level)
    print(f"Input:  {input_path}")
    print(f"Tree:   {tree_name}")
    print(f"Events: {tree.num_entries}")
    print(f"Branches: {len(all_keys)}")
    print(f"Compression: {compression}({level})")

    n_written = 0
    with uproot.recreate(output_path, compression=comp) as out_file:
        out_tree = None

        for chunk in tree.iterate(library="ak", step_size=chunk_size):
            n_chunk = len(chunk[all_keys[0]])
            out_data = {k: chunk[k] for k in all_keys}

            if out_tree is None:
                out_tree = out_file.mktree(
                    tree_name, {k: v.type for k, v in out_data.items()}
                )

            out_tree.extend(out_data)
            n_written += n_chunk
            print(f"  {n_written}/{tree.num_entries} events", end="\r")

    print(f"\nOutput: {output_path}")
    print(f"Written {n_written} events")


def main():
    parser = argparse.ArgumentParser(description="Convert RNTuple to TTree")
    parser.add_argument("input", help="Input ROOT file (RNTuple)")
    parser.add_argument("output", help="Output ROOT file (TTree)")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple", help="Tree name")
    parser.add_argument("--chunk", type=int, default=100, help="Chunk size for reading")
    parser.add_argument(
        "--compression", choices=["zstd", "zlib", "lz4"], default="zstd"
    )
    parser.add_argument("--level", type=int, default=1, help="Compression level")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    convert(
        args.input,
        args.output,
        tree_name=args.tree,
        chunk_size=args.chunk,
        compression=args.compression,
        level=args.level,
    )


if __name__ == "__main__":
    main()
