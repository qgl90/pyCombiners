"""Unit tests for JSON input loader helpers."""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import json
import tempfile
import unittest
from pathlib import Path

from trackcomb import ParticleHypothesis
from trackcomb.io import load_events_json, load_mass_hypotheses_json


class TestIOLoaders(unittest.TestCase):
    """Validate parsing for event-batch and named-hypothesis JSON inputs."""

    def test_load_mass_hypotheses_supports_named_and_custom_entries(self) -> None:
        """Mass loader should parse string aliases, pid aliases, and custom masses."""
        payload = {
            "mass_hypotheses": [
                ["pi", {"pid": "kaon"}, {"name": "customX", "mass": 1.234}],
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "masses.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            [hyp_set] = load_mass_hypotheses_json(path)
        self.assertIsInstance(hyp_set[0], ParticleHypothesis)
        self.assertIsInstance(hyp_set[1], ParticleHypothesis)
        self.assertIsInstance(hyp_set[2], ParticleHypothesis)
        hyp0 = hyp_set[0]
        hyp1 = hyp_set[1]
        hyp2 = hyp_set[2]
        assert isinstance(hyp0, ParticleHypothesis)
        assert isinstance(hyp1, ParticleHypothesis)
        assert isinstance(hyp2, ParticleHypothesis)
        self.assertEqual(hyp0.name, "pi")
        self.assertEqual(hyp1.name, "K")
        self.assertEqual(hyp2.name, "customX")
        self.assertAlmostEqual(hyp2.mass, 1.234, places=12)

    def test_load_events_json_parses_event_payload(self) -> None:
        """Event loader should parse per-event track and PV containers."""
        payload = {
            "events": [
                {
                    "event_id": "evt42",
                    "tracks": [
                        {
                            "track_id": "t0",
                            "z": 0.0,
                            "state": {"x": 0.0, "y": 0.0, "tx": 0.01, "ty": 0.0, "time": 1.0},
                            "cov4": [
                                [0.01, 0.0, 0.0, 0.0],
                                [0.0, 0.01, 0.0, 0.0],
                                [0.0, 0.0, 0.001, 0.0],
                                [0.0, 0.0, 0.0, 0.001],
                            ],
                            "sigma_time": 0.1,
                            "p": 5.0,
                            "charge": 1,
                        }
                    ],
                    "primary_vertices": [
                        {
                            "pv_id": "pv0",
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "cov3": [
                                [0.01, 0.0, 0.0],
                                [0.0, 0.01, 0.0],
                                [0.0, 0.0, 0.01],
                            ],
                            "time": 1.0,
                            "sigma_time": 0.05,
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            [event] = load_events_json(path)
        self.assertEqual(event.event_id, "evt42")
        self.assertEqual(len(event.tracks), 1)
        self.assertEqual(len(event.primary_vertices), 1)
        self.assertEqual(event.tracks[0].track_id, "t0")
        self.assertEqual(event.tracks[0].source_track_ids, ("t0",))
        self.assertEqual(event.primary_vertices[0].pv_id, "pv0")


if __name__ == "__main__":
    unittest.main()
