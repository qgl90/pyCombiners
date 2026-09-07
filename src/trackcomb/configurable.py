"""Decorator for functions with layered parameter overrides."""

from __future__ import annotations

import contextvars
import functools
import threading
from contextlib import contextmanager

# Per-thread/coroutine dynamic scope
_CFG = contextvars.ContextVar("configurable_bindings", default={})

# Process-level global defaults
_GLOBAL = {}
_GLOBAL_LOCK = threading.RLock()


def configurable(fn):
    """Make a function's keyword args overridable via bind() and global_bind()."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # global defaults
        with _GLOBAL_LOCK:
            merged = dict(_GLOBAL.get(fn, {}))
        # dynamic bind layers
        for layer in _CFG.get().get(fn, ()):
            merged.update(layer)
        # explicit kwargs win
        merged.update(kwargs)
        return fn(*args, **merged)

    @contextmanager
    def bind(**overrides):
        """Context manager: temporarily override kwargs."""
        cur = _CFG.get()
        new = dict(cur)
        new[fn] = cur.get(fn, ()) + (overrides,)
        token = _CFG.set(new)
        try:
            yield wrapper
        finally:
            _CFG.reset(token)

    def global_bind(**overrides):
        """Set process-level defaults (persistent)."""
        with _GLOBAL_LOCK:
            prev = _GLOBAL.get(fn)
            now = dict(prev) if prev else {}
            now.update(overrides)
            _GLOBAL[fn] = now

    def reset_global():
        """Clear all global defaults for this function."""
        with _GLOBAL_LOCK:
            _GLOBAL.pop(fn, None)

    def partial(**overrides):
        """Return a new callable with kwargs pre-bound."""

        @functools.wraps(fn)
        def bound(*args, **kwargs):
            merged = dict(overrides)
            merged.update(kwargs)
            return wrapper(*args, **merged)

        # Propagate configurable interface
        bound.bind = bind
        bound.partial = lambda **kw: partial(**{**overrides, **kw})
        bound.global_bind = global_bind
        bound.reset_global = reset_global
        return bound

    wrapper.bind = bind
    wrapper.partial = partial
    wrapper.global_bind = global_bind
    wrapper.reset_global = reset_global
    return wrapper
