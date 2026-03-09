"""Singleton ONNX model cache."""

from __future__ import annotations

import numpy as np


class OnnxModel:
    """Wrapper around an ONNX session with a simple run() interface."""

    def __init__(self, path: str):
        import onnxruntime as ort

        self._session = ort.InferenceSession(path)
        self._input_name = self._session.get_inputs()[0].name

    def run(self, features: np.ndarray) -> np.ndarray:
        return self._session.run(
            None, {self._input_name: features.astype(np.float32)}
        )[0].flatten()


_cache: dict[str, OnnxModel] = {}


def onnx_models(path: str) -> OnnxModel:
    """Get a cached ONNX model by path."""
    if path not in _cache:
        _cache[path] = OnnxModel(path)
    return _cache[path]
