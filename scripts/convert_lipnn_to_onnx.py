#!/usr/bin/env python3
"""Convert Allen/HLT1 Lipschitz NN (JSON) to ONNX.

Architecture: FC layers with GroupSort2 activation, monotonicity via
sigma * dot(input, constraints), final sigmoid.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _add_groupsort2(nodes, initializers, input_name, output_name, width):
    """Emit ONNX ops for GroupSort2: sort adjacent pairs (min, max)."""
    pfx = output_name
    even_idx = np.arange(0, width, 2, dtype=np.int64)
    odd_idx = np.arange(1, width, 2, dtype=np.int64)

    initializers.extend(
        [
            numpy_helper.from_array(even_idx, name=f"{pfx}_even_idx"),
            numpy_helper.from_array(odd_idx, name=f"{pfx}_odd_idx"),
            numpy_helper.from_array(
                np.array([2], dtype=np.int64), name=f"{pfx}_unsq_axes"
            ),
            numpy_helper.from_array(
                np.array([-1, width], dtype=np.int64), name=f"{pfx}_shape"
            ),
        ]
    )

    # Gather even/odd elements along feature axis
    nodes.append(
        helper.make_node(
            "Gather", [input_name, f"{pfx}_even_idx"], [f"{pfx}_even"], axis=1
        )
    )
    nodes.append(
        helper.make_node(
            "Gather", [input_name, f"{pfx}_odd_idx"], [f"{pfx}_odd"], axis=1
        )
    )

    # Element-wise min / max
    nodes.append(
        helper.make_node("Min", [f"{pfx}_even", f"{pfx}_odd"], [f"{pfx}_min"])
    )
    nodes.append(
        helper.make_node("Max", [f"{pfx}_even", f"{pfx}_odd"], [f"{pfx}_max"])
    )

    # Interleave: unsqueeze → concat → reshape
    nodes.append(
        helper.make_node(
            "Unsqueeze", [f"{pfx}_min", f"{pfx}_unsq_axes"], [f"{pfx}_min_us"]
        )
    )
    nodes.append(
        helper.make_node(
            "Unsqueeze", [f"{pfx}_max", f"{pfx}_unsq_axes"], [f"{pfx}_max_us"]
        )
    )
    nodes.append(
        helper.make_node(
            "Concat",
            [f"{pfx}_min_us", f"{pfx}_max_us"],
            [f"{pfx}_cat"],
            axis=2,
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", [f"{pfx}_cat", f"{pfx}_shape"], [output_name]
        )
    )


def convert(json_path, onnx_path):
    """Read JSON model and write ONNX file."""
    with open(json_path) as f:
        data = json.load(f)

    sigma = float(data["sigmanet.sigma"][0])
    nominal_cut = float(data["nominal_cut"])
    suffix = "_orig" if "sigmanet.nn.0.weight_orig" in data else ""

    # Discover layer indices (may be 0,2,4 or 0,1,2 depending on model)
    layer_indices = sorted(
        int(k.split(".")[2]) for k in data if k.endswith(f"weight{suffix}")
    )

    n_input = len(data[f"sigmanet.nn.{layer_indices[0]}.weight{suffix}"][0])

    constraints = np.array(
        data.get("constraints", [1.0, 1.0, 0.0, 1.0][:n_input]),
        dtype=np.float32,
    )

    nodes = []
    initializers = []
    current = "input"

    for i, layer_idx in enumerate(layer_indices):
        W = np.array(
            data[f"sigmanet.nn.{layer_idx}.weight{suffix}"], dtype=np.float32
        )
        b = np.array(data[f"sigmanet.nn.{layer_idx}.bias"], dtype=np.float32)
        is_last = i == len(layer_indices) - 1

        # W is (out_features, in_features); transpose for x @ W^T
        initializers.append(numpy_helper.from_array(W.T, name=f"W{i}"))
        initializers.append(numpy_helper.from_array(b, name=f"b{i}"))

        nodes.append(
            helper.make_node("MatMul", [current, f"W{i}"], [f"mm{i}"])
        )
        nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"lin{i}"]))

        if not is_last:
            _add_groupsort2(
                nodes, initializers, f"lin{i}", f"gs{i}", W.shape[0]
            )
            current = f"gs{i}"
        else:
            current = f"lin{i}"

    # Monotonicity: output += sigma * (input @ constraints)
    initializers.append(
        numpy_helper.from_array(constraints.reshape(-1, 1), name="constraints")
    )
    initializers.append(
        numpy_helper.from_array(
            np.array([sigma], dtype=np.float32), name="sigma"
        )
    )

    nodes.append(
        helper.make_node("MatMul", ["input", "constraints"], ["mono_dot"])
    )
    nodes.append(helper.make_node("Mul", ["mono_dot", "sigma"], ["mono"]))
    nodes.append(helper.make_node("Add", [current, "mono"], ["pre_sigmoid"]))
    nodes.append(helper.make_node("Sigmoid", ["pre_sigmoid"], ["output"]))

    # Build graph and model
    X = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, n_input]
    )
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    graph = helper.make_graph(
        nodes, "LipschitzNN", [X], [Y], initializer=initializers
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)

    layer_sizes = [n_input] + [
        len(data[f"sigmanet.nn.{idx}.bias"]) for idx in layer_indices
    ]
    print(f"Converted {json_path} -> {onnx_path}")
    print(f"  architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"  sigma={sigma}, nominal_cut={nominal_cut}")
    print(f"  constraints={constraints.tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input JSON model file")
    p.add_argument("output", help="Output ONNX model file")
    args = p.parse_args()
    convert(args.input, args.output)
