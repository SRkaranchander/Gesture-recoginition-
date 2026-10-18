"""
security/security_feedback.py
-------------------------------
Security Feedback Loop.

Aggregates transition outcomes and feeds them back into the security context
so that future authorization decisions reflect the session's security history.

Architecture:
    Attempt → Decision → Outcome → Security event → Update context
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from security.security_escalation import SecurityEscalation

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    timestamp: float
    event_type: str   # ALLOW | DENY | FAIL_BIO | IDENTITY_LOSS | LOCKOUT | ANOMALY
    detail: str = ""


class SecurityFeedback:
    """
    Receives outcome events and forwards them to the SecurityEscalation engine.
    Also maintains a short-term event history for metrics export.
    """

    def __init__(self, escalation: SecurityEscalation) -> None:
        self._esc = escalation
        self._history: list[FeedbackEvent] = []

    def _record(self, event_type: str, detail: str = "") -> None:
        self._history.append(FeedbackEvent(time.time(), event_type, detail))
        if len(self._history) > 500:
            self._history = self._history[-500:]

    def on_allow(self, gesture: str) -> None:
        self._record("ALLOW", gesture)
        self._esc.record_successful_transition()

    def on_deny(self, reason: str) -> None:
        self._record("DENY", reason)
        self._esc.record_repeated_denial()

    def on_failed_biometric(self, conf: float) -> None:
        self._record("FAIL_BIO", f"conf={conf:.3f}")
        self._esc.record_failed_auth()

    def on_identity_loss(self) -> None:
        self._record("IDENTITY_LOSS")
        self._esc.record_identity_loss()

    def on_lockout(self) -> None:
        self._record("LOCKOUT")
        self._esc.record_lockout()

    def on_gesture_anomaly(self, score: float) -> None:
        self._record("ANOMALY", f"auth_score={score:.3f}")
        self._esc.record_gesture_anomaly()

    def on_successful_auth(self) -> None:
        self._record("AUTH_OK")
        self._esc.record_successful_auth()

    def export_metrics(self) -> dict:
        """Return aggregated counts for experiment export."""
        counts: dict[str, int] = {}
        for ev in self._history:
            counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
        return counts
