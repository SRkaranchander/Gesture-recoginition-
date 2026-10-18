"""
security/temporal_binding.py
------------------------------
Temporal Biometric-Gesture Binding.

Tracks whether the authenticated identity remained continuously associated
with the gesture execution interval.  If the face disappears and a different
(or no) identity performs the gesture, the binding score drops, which feeds
into the adaptive authorization score.

Security invariant: an old authorization window cannot authorize a gesture
if identity continuity has been broken during the gesture window.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ADAPTIVE_SECURITY_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class BindingRecord:
    """Snapshot of identity state at a point in time."""
    timestamp:  float
    user_id:    Optional[int]
    confidence: float


class TemporalBinding:
    """
    Maintains a rolling window of identity observations and computes a
    TemporalBindingScore representing how continuously the authenticated
    user was present during the gesture window.

    Score = fraction of observations where the correct user was detected,
            penalised by the longest continuous gap.
    """

    def __init__(self) -> None:
        cfg = ADAPTIVE_SECURITY_CONFIG["temporal_binding"]
        self._max_gap: float = cfg.get("max_identity_gap", 1.0)
        self._window:  list[BindingRecord] = []
        self._window_sec: float = 5.0   # keep last N seconds of observations
        self._authenticated_user: Optional[int] = None
        self._gesture_start: Optional[float] = None

    # ── Control ────────────────────────────────────────────────────────────────

    def set_authenticated_user(self, user_id: Optional[int]) -> None:
        self._authenticated_user = user_id

    def start_gesture(self) -> None:
        """Call when a gesture begins accumulating stability frames."""
        self._gesture_start = time.time()

    def record_observation(self, user_id: Optional[int], confidence: float) -> None:
        """Call every frame with the current face-verification result."""
        now = time.time()
        self._window.append(BindingRecord(now, user_id, confidence))
        # Prune old observations
        cutoff = now - self._window_sec
        self._window = [r for r in self._window if r.timestamp >= cutoff]

    # ── Score ──────────────────────────────────────────────────────────────────

    def binding_score(self) -> float:
        """
        Return a score in [0, 1] representing identity continuity.

        1.0 = authenticated user was present in every observation.
        0.0 = authenticated user was never present, or a gap exceeded max_gap.
        """
        if not ADAPTIVE_SECURITY_CONFIG["temporal_binding"]["enabled"]:
            return 1.0

        if self._authenticated_user is None:
            return 0.0

        # Only consider observations since gesture started
        start = self._gesture_start or (time.time() - self._window_sec)
        relevant = [r for r in self._window if r.timestamp >= start]

        if not relevant:
            return 0.5   # no data — neutral

        correct = sum(1 for r in relevant if r.user_id == self._authenticated_user)
        ratio   = correct / len(relevant)

        # Penalise for any gap exceeding max_gap
        gap_penalty = self._max_gap_penalty(relevant)

        score = ratio * (1.0 - gap_penalty)
        return float(max(0.0, min(1.0, score)))

    def _max_gap_penalty(self, records: list[BindingRecord]) -> float:
        """Return a penalty [0,1] based on the longest identity-absent gap."""
        if len(records) < 2:
            return 0.0
        max_gap = 0.0
        for i in range(1, len(records)):
            if records[i].user_id != self._authenticated_user:
                gap = records[i].timestamp - records[i - 1].timestamp
                max_gap = max(max_gap, gap)
        if max_gap >= self._max_gap:
            return 1.0
        return max_gap / self._max_gap

    def identity_lost(self) -> bool:
        """Return True if the last observation shows no authenticated user."""
        if not self._window:
            return False
        return self._window[-1].user_id != self._authenticated_user

    def reset(self) -> None:
        self._window.clear()
        self._gesture_start = None
