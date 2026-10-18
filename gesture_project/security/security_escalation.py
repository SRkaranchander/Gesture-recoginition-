"""
security/security_escalation.py
---------------------------------
Security Escalation State Machine.

Tracks suspicious events and dynamically adjusts the security level.
Higher levels impose stricter biometric thresholds and shorter auth windows.

Levels:
    0 = NORMAL
    1 = SUSPICIOUS
    2 = ELEVATED
    3 = RESTRICTED
    4 = LOCKED

De-escalation occurs automatically after a configurable period of clean activity.
"""

import time
import logging
from typing import Optional

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ADAPTIVE_SECURITY_CONFIG

logger = logging.getLogger(__name__)

LEVEL_NAMES = {0: "NORMAL", 1: "SUSPICIOUS", 2: "ELEVATED", 3: "RESTRICTED", 4: "LOCKED"}


class SecurityEscalation:
    """
    Maintains a security level that rises on suspicious events and
    falls after a configurable period of legitimate activity.
    """

    def __init__(self) -> None:
        cfg = ADAPTIVE_SECURITY_CONFIG["security_escalation"]
        self._level: int = 0
        self._last_event_time: float = time.time()
        self._last_clean_time: float = time.time()
        self._deescalate_after: float = cfg.get("deescalate_after_sec", 120.0)
        self._enabled: bool = cfg.get("enabled", True)

    # ── Events that raise the level ────────────────────────────────────────────

    def record_failed_auth(self) -> None:
        self._escalate(1, "failed authentication")

    def record_identity_loss(self) -> None:
        self._escalate(1, "identity loss")

    def record_repeated_denial(self) -> None:
        self._escalate(1, "repeated transition denial")

    def record_gesture_anomaly(self) -> None:
        self._escalate(1, "gesture authenticity anomaly")

    def record_lockout(self) -> None:
        self._set_level(4, "lockout triggered")

    # ── Events that reset / lower the level ───────────────────────────────────

    def record_successful_auth(self) -> None:
        self._last_clean_time = time.time()
        if self._level > 0:
            self._set_level(max(0, self._level - 1), "successful authentication")

    def record_successful_transition(self) -> None:
        self._last_clean_time = time.time()
        self._maybe_deescalate()

    # ── Periodic de-escalation ─────────────────────────────────────────────────

    def tick(self) -> None:
        """Call periodically (e.g., every frame) to allow time-based de-escalation."""
        self._maybe_deescalate()

    def _maybe_deescalate(self) -> None:
        if not self._enabled or self._level == 0:
            return
        if time.time() - self._last_clean_time >= self._deescalate_after:
            self._set_level(max(0, self._level - 1), "time-based de-escalation")
            self._last_clean_time = time.time()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _escalate(self, by: int, reason: str) -> None:
        if not self._enabled:
            return
        new_level = min(4, self._level + by)
        self._set_level(new_level, reason)
        self._last_event_time = time.time()

    def _set_level(self, level: int, reason: str) -> None:
        if level != self._level:
            logger.warning("Security level: %s → %s (%s)",
                           LEVEL_NAMES[self._level], LEVEL_NAMES[level], reason)
            self._level = level

    # ── Read-only ──────────────────────────────────────────────────────────────

    @property
    def level(self) -> int:
        return self._level

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self._level, "UNKNOWN")

    def is_locked(self) -> bool:
        return self._level >= 4
