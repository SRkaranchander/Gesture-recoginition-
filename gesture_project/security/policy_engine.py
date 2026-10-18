"""
security/policy_engine.py
--------------------------
Determines whether biometric authentication is required for a given
TransitionCandidate based on the configured risk policy.

The policy is read from config/settings.py — no values are hard-coded here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import RISK_POLICY
from state.transition_manager import TransitionCandidate


def biometric_required(candidate: TransitionCandidate) -> bool:
    """
    Return True if biometric authentication must be performed before
    this transition is authorized.

    Policy rules (from settings.RISK_POLICY):
      NEVER       → never require biometric regardless of transition flag
      ALWAYS      → always require biometric regardless of transition flag
      CONFIGURABLE → follow the biometric_required flag on the transition row
    """
    policy = RISK_POLICY.get(candidate.risk_level, "ALWAYS")

    if policy == "NEVER":
        return False
    if policy == "ALWAYS":
        return True
    # CONFIGURABLE — defer to the transition table
    return candidate.biometric_required


def risk_colour(risk_level: str) -> tuple[int, int, int]:
    """Return a BGR colour for UI display of a risk level."""
    return {
        "LOW":      (0, 200, 80),
        "MEDIUM":   (0, 180, 220),
        "HIGH":     (0, 80, 220),
        "CRITICAL": (0, 0, 200),
    }.get(risk_level, (200, 200, 200))
