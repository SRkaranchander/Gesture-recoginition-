"""
audit_log/audit_logger.py
--------------------------
Extended audit logger — accepts both original and new adaptive fields.
Old call sites continue to work unchanged (all new fields are optional).
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database import db

logger = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    def log(
        self,
        *,
        user_id: Optional[int],
        previous_state: str,
        gesture: str,
        gesture_confidence: float,
        destination_state: str,
        risk_level: str,
        biometric_required: bool,
        biometric_type: Optional[str],
        biometric_confidence: Optional[float],
        authentication_result: Optional[str],
        authorization_result: str,
        final_state: str,
        action: str,
        reason: Optional[str] = None,
        # ── Adaptive fields (all optional for backward compatibility) ──────────
        authorization_score: Optional[float] = None,
        gesture_authenticity_score: Optional[float] = None,
        temporal_binding_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        biometric_threshold_used: Optional[float] = None,
        authorization_threshold: Optional[float] = None,
        authorization_level: Optional[str] = None,
        security_state: Optional[str] = None,
        security_escalation_level: Optional[int] = None,
        token_id: Optional[str] = None,
        authentication_age: Optional[float] = None,
        identity_continuity: Optional[float] = None,
        decision_reason: Optional[str] = None,
    ) -> None:
        entry: dict = {
            "timestamp":            datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
            "session_id":           self._session_id,
            "user_id":              user_id,
            "previous_state":       previous_state,
            "gesture":              gesture,
            "gesture_confidence":   round(gesture_confidence, 4),
            "destination_state":    destination_state,
            "risk_level":           risk_level,
            "biometric_required":   int(biometric_required),
            "biometric_type":       biometric_type,
            "biometric_confidence": round(biometric_confidence, 4) if biometric_confidence is not None else None,
            "authentication_result": authentication_result,
            "authorization_result":  authorization_result,
            "final_state":          final_state,
            "action":               action,
            "reason":               reason,
        }
        # Only include adaptive fields when provided
        optional = {
            "authorization_score":       authorization_score,
            "gesture_authenticity_score":gesture_authenticity_score,
            "temporal_binding_score":    temporal_binding_score,
            "risk_score":                risk_score,
            "biometric_threshold_used":  biometric_threshold_used,
            "authorization_threshold":   authorization_threshold,
            "authorization_level":       authorization_level,
            "security_state":            security_state,
            "security_escalation_level": security_escalation_level,
            "token_id":                  token_id,
            "authentication_age":        authentication_age,
            "identity_continuity":       identity_continuity,
            "decision_reason":           decision_reason,
        }
        entry.update({k: v for k, v in optional.items() if v is not None})
        db.insert_log(entry)
        logger.info(
            "[AUDIT] %s | %s->%s | gesture=%s(%.2f) | bio=%s(%.2f) | %s | %s | score=%.2f",
            previous_state, gesture, final_state,
            gesture, gesture_confidence,
            authentication_result, biometric_confidence or 0.0,
            authorization_result, action,
            authorization_score or 0.0,
        )
