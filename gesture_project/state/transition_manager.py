"""
state/transition_manager.py
----------------------------
Looks up the transition table and returns a structured result.
Does NOT make authorization decisions — that is the policy engine's job.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database import db


@dataclass
class TransitionCandidate:
    """Everything the policy engine needs to make an authorization decision."""
    source_state: str
    gesture: str
    destination_state: str
    risk_level: str
    biometric_required: bool
    biometric_type: str
    confidence_threshold: float
    authorization_timeout: float
    action: str


def find_transition(
    source_state: str,
    gesture: str,
    gesture_confidence: float,
) -> Optional[TransitionCandidate]:
    """
    Look up a valid, enabled transition for (source_state, gesture).
    Returns None if no transition exists or confidence is below the row threshold.
    """
    row = db.lookup_transition(source_state, gesture)
    if row is None:
        return None

    if gesture_confidence < row["confidence_threshold"]:
        return None

    return TransitionCandidate(
        source_state=row["source_state"],
        gesture=row["gesture"],
        destination_state=row["destination_state"],
        risk_level=row["risk_level"],
        biometric_required=bool(row["biometric_required"]),
        biometric_type=row["biometric_type"],
        confidence_threshold=row["confidence_threshold"],
        authorization_timeout=row["timeout"],
        action=row["action"],
    )
