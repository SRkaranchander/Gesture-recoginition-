"""
security/gesture_authenticity.py
----------------------------------
Gesture Authenticity Scorer.

Operates alongside the existing gesture classifier without replacing it.
Uses temporal features (velocity, duration, trajectory consistency) to
detect abnormal, replayed, or inconsistent gesture patterns.

Returns a GestureAuthenticityScore in [0, 1].
1.0 = gesture appears natural and consistent.
0.0 = gesture appears anomalous.
"""

import time
import numpy as np
import logging
from typing import Optional

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ADAPTIVE_SECURITY_CONFIG

logger = logging.getLogger(__name__)


class GestureAuthenticity:
    """
    Heuristic authenticity detector based on:
      - gesture duration (too short = suspicious)
      - wrist velocity variance (too uniform = possibly replayed)
      - landmark trajectory consistency
    """

    def __init__(self) -> None:
        cfg = ADAPTIVE_SECURITY_CONFIG["gesture_authenticity"]
        self._min_duration: float = cfg.get("min_duration_sec", 0.20)
        self._max_vel_change: float = cfg.get("max_velocity_change", 3.0)
        self._enabled: bool = cfg.get("enabled", True)
        self._start_time: Optional[float] = None
        self._frame_buffer: list[np.ndarray] = []   # list of 78-D feature vectors

    def start(self) -> None:
        """Call when a new gesture begins accumulating."""
        self._start_time = time.time()
        self._frame_buffer.clear()

    def add_frame(self, feature_vector: np.ndarray) -> None:
        """Add one 78-D feature vector from the existing pipeline."""
        self._frame_buffer.append(feature_vector.copy())

    def score(self) -> float:
        """
        Compute and return the authenticity score.
        Returns 1.0 if the module is disabled.
        """
        if not self._enabled:
            return 1.0

        if self._start_time is None or len(self._frame_buffer) < 4:
            return 0.8   # insufficient data — mildly penalise

        duration = time.time() - self._start_time
        dur_score = min(1.0, duration / max(self._min_duration, 0.01))

        # Wrist position is the first 3 values of the normalised landmark vector
        # (after wrist-origin normalisation they are ~0, but velocity is non-zero)
        wrist_positions = np.array([fv[:3] for fv in self._frame_buffer])
        velocities = np.diff(wrist_positions, axis=0)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)

        if len(vel_magnitudes) < 2:
            vel_score = 1.0
        else:
            vel_std  = float(np.std(vel_magnitudes))
            vel_mean = float(np.mean(vel_magnitudes)) + 1e-8
            cv = vel_std / vel_mean   # coefficient of variation
            # Natural gestures have moderate variation; perfectly uniform = suspicious
            if cv < 0.05:
                vel_score = 0.5   # suspiciously uniform
            elif cv > self._max_vel_change:
                vel_score = 0.6   # erratic
            else:
                vel_score = 1.0

        # Trajectory consistency: check that joint angles evolve smoothly
        angles = np.array([fv[63:] for fv in self._frame_buffer])  # last 15 dims
        angle_diffs = np.abs(np.diff(angles, axis=0))
        max_jump = float(np.max(angle_diffs)) if angle_diffs.size > 0 else 0.0
        traj_score = max(0.0, 1.0 - max_jump / 2.0)

        final = 0.35 * dur_score + 0.35 * vel_score + 0.30 * traj_score
        logger.debug("GestureAuthenticity: dur=%.2f vel=%.2f traj=%.2f → %.3f",
                     dur_score, vel_score, traj_score, final)
        return float(max(0.0, min(1.0, final)))

    def reset(self) -> None:
        self._start_time = None
        self._frame_buffer.clear()
