"""event_info: EventInfo branches -> flat numpy arrays (one per event)."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..io import read
from ..models import Container


def load_event_info(chunk: Container) -> dict:
    return {
        "run_number": ak.to_numpy(read(chunk, "EventInfo/RunNumber")).astype(
            np.int64
        ),
        "event_number": ak.to_numpy(
            read(chunk, "EventInfo/EventNumber")
        ).astype(np.int64),
        "bunch_crossing_id": ak.to_numpy(
            read(chunk, "EventInfo/BunchCrossingID")
        ).astype(np.int64),
    }
