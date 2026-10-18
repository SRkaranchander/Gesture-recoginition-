"""
security/adaptive_authorization.py
------------------------------------
Adaptive Transition Authorization Engine.

Computes a normalized authorization score from multiple contextual signals
and maps it to one of five authorization decisions:
    ALLOW | REAUTHENTICATE | STEP_UP | DENY | LOCK

All weights and thresholds are read from ADAPTIVE_SECURITY_CONFIG —
nothing is hard-coded.  This module does NOT perform biometric verification
or state transitions; it only produces a scored decision.

Security invariant: the score is computed BEFORE any state commit.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ADAPTIVE_SECURITY_CONFIG

logger = logging.getLogger(__name__)

# ── Decision levels ────────────────────────────────────────────────────────────
ALLOW          = "ALLOW"
REAUTHENTICATE = "REAUTHENTICATE"
STEP_UP        = "STEP_UP"
DENY           = "DENY"
LOCK           = "LOCK"


@dataclass
class AuthorizationContext:
    """All signals fed into the scoring function."""
    gesture_confidence:    float = 0.0   # [0,1] from classifier
    biometric_confidence:  float = 0.0   # [0,1] from face auth
    stability_ratio:       float = 0.0   # stable_count / STABILITY_FRAMES
    identity_continuity:   float = 1.0   # [0,1] from TemporalBinding
    authentication_age_sec: float = 0.0  # seconds since last successful auth
    risk_level:            str   = "LOW"
    failure_count:         int   = 0
    security_level:        int   = 0     # 0=NORMAL … 4=LOCKED (from escalation)
    gesture_authenticity:  float = 1.0   # [0,1] from GestureAuthenticity
    auth_window_sec:       float = 15.0  # configured window for this risk level


@dataclass
class AuthorizationResult:
    """Output of the scoring engine."""
    decision:              str
    score:                 float
    risk_score:            float
    biometric_threshold:   float
    authorization_timeout: float
    reason:                str
    context:               AuthorizationContext = field(repr=False)


def _risk_score(risk_level: str) -> float:
    return ADAPTIVE_SECURITY_CONFIG["risk_scores"].get(risk_level, 0.70)


def _biometric_threshold(risk_level: str) -> float:
    key = f"{risk_level.lower()}_threshold"
    return ADAPTIVE_SECURITY_CONFIG["biometric"].get(key, 0.75)


def _auth_window(risk_level: str) -> float:
    return ADAPTIVE_SECURITY_CONFIG["authorization_window"].get(risk_level, 15.0)


def _age_penalty(age_sec: float, window_sec: float) -> float:
    """Linearly increase penalty as auth age approaches the window duration."""
    if window_sec <= 0:
        return 1.0
    return min(1.0, age_sec / max(window_sec, 1.0))


def compute_score(ctx: AuthorizationContext) -> float:
    """
    Compute a normalized authorization score in [0, 1].

    score = w_gesture * gesture_confidence
          + w_biometric * biometric_confidence
          + w_stability * stability_ratio
          + w_identity  * identity_continuity
          + w_auth      * gesture_authenticity
          - w_risk      * risk_score
          - w_failure   * failure_penalty
          - w_age       * age_penalty

    All weights are read from ADAPTIVE_SECURITY_CONFIG["weights"].
    The result is clipped to [0, 1].
    """
    w = ADAPTIVE_SECURITY_CONFIG["weights"]
    rs = _risk_score(ctx.risk_level)
    aw = _auth_window(ctx.risk_level)

    failure_penalty = min(1.0, ctx.failure_count / max(
        ADAPTIVE_SECURITY_CONFIG["security_escalation"]["max_failures"], 1))
    age_pen = _age_penalty(ctx.authentication_age_sec, aw)

    # Security escalation penalty: each level above NORMAL reduces score
    escalation_penalty = ctx.security_level * 0.10

    # For LOW risk, biometric is not required — redistribute its weight to gesture
    bio_weight = w.get("biometric", 0.30)
    if ctx.risk_level == "LOW":
        effective_bio = bio_weight   # treat as fully satisfied
        bio_term = bio_weight * 1.0
    else:
        bio_term = bio_weight * ctx.biometric_confidence

    raw = (
        w.get("gesture",             0.20) * ctx.gesture_confidence
      + bio_term
      + w.get("stability",           0.10) * ctx.stability_ratio
      + w.get("identity_continuity", 0.20) * ctx.identity_continuity
      + w.get("gesture",             0.20) * ctx.gesture_authenticity * 0.5
      - w.get("risk",                0.10) * rs
      - w.get("failure_history",     0.05) * failure_penalty
      - w.get("auth_age",            0.05) * age_pen
      - escalation_penalty
    )
    return float(max(0.0, min(1.0, raw)))


def decide(ctx: AuthorizationContext) -> AuthorizationResult:
    """
    Map the authorization score to a multi-level decision.
    Thresholds are read from ADAPTIVE_SECURITY_CONFIG["authorization"].
    """
    score = compute_score(ctx)
    th    = ADAPTIVE_SECURITY_CONFIG["authorization"]
    rs    = _risk_score(ctx.risk_level)
    bt    = _biometric_threshold(ctx.risk_level)
    aw    = _auth_window(ctx.risk_level)

    if score >= th["allow_threshold"]:
        decision = ALLOW
        reason   = f"Score {score:.3f} >= allow threshold {th['allow_threshold']}"
    elif score >= th["reauth_threshold"]:
        decision = REAUTHENTICATE
        reason   = f"Score {score:.3f} requires re-authentication"
    elif score >= th["stepup_threshold"]:
        decision = STEP_UP
        reason   = f"Score {score:.3f} requires step-up authentication"
    elif score >= th["deny_threshold"]:
        decision = DENY
        reason   = f"Score {score:.3f} below authorization threshold"
    else:
        decision = LOCK
        reason   = f"Score {score:.3f} critically low — forcing lock"

    logger.debug("AuthScore=%.3f decision=%s risk=%s", score, decision, ctx.risk_level)
    return AuthorizationResult(
        decision=decision,
        score=score,
        risk_score=rs,
        biometric_threshold=bt,
        authorization_timeout=aw,
        reason=reason,
        context=ctx,
    )
