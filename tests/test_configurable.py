"""Tests for the configurable decorator."""

from __future__ import annotations

import threading
import unittest

from trackcomb.configurable import configurable


@configurable
def add(a, b=10, c=0):
    return a + b + c


class TestConfigurable(unittest.TestCase):
    def setUp(self):
        add.reset_global()

    def test_default_kwargs(self):
        self.assertEqual(add(1), 11)

    def test_explicit_kwargs(self):
        self.assertEqual(add(1, b=20), 21)

    def test_bind_overrides_default(self):
        with add.bind(b=100) as f:
            self.assertEqual(f(1), 101)
        # outside bind, back to default
        self.assertEqual(add(1), 11)

    def test_bind_explicit_wins(self):
        with add.bind(b=100) as f:
            self.assertEqual(f(1, b=5), 6)

    def test_bind_nested(self):
        with add.bind(b=100) as f1:
            with f1.bind(c=1) as f2:
                # b=100 from outer, c=1 from inner
                self.assertEqual(f2(0), 101)
            # inner gone, c back to 0
            self.assertEqual(f1(0), 100)

    def test_bind_inner_overrides_outer(self):
        with add.bind(b=100) as f1:
            with f1.bind(b=200) as f2:
                self.assertEqual(f2(0), 200)
            self.assertEqual(f1(0), 100)

    def test_global_bind_persistent(self):
        add.global_bind(b=42)
        self.assertEqual(add(1), 43)

    def test_global_bind_stacks(self):
        add.global_bind(b=42)
        add.global_bind(c=5)
        self.assertEqual(add(0), 47)  # b=42, c=5

    def test_global_bind_with_local_bind(self):
        add.global_bind(b=50)
        with add.bind(c=1) as f:
            # global b=50, bind c=1
            self.assertEqual(f(0), 51)

    def test_bind_overrides_global(self):
        add.global_bind(b=50)
        with add.bind(b=99) as f:
            self.assertEqual(f(0), 99)

    def test_reset_global(self):
        add.global_bind(b=42)
        add.reset_global()
        self.assertEqual(add(1), 11)

    def test_thread_isolation(self):
        """bind() in one thread does not leak to another."""
        results = {}

        def worker(name, use_bind):
            if use_bind:
                with add.bind(b=999) as f:
                    results[name] = f(0)
            else:
                results[name] = add(0)

        t1 = threading.Thread(target=worker, args=("bound", True))
        t2 = threading.Thread(target=worker, args=("default", False))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(results["bound"], 999)
        self.assertEqual(results["default"], 10)

    def test_global_visible_across_threads(self):
        results = {}

        def worker():
            results["val"] = add(0)

        add.global_bind(b=77)
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(results["val"], 77)

    def test_wraps_preserves_metadata(self):
        self.assertEqual(add.__name__, "add")


if __name__ == "__main__":
    unittest.main()
