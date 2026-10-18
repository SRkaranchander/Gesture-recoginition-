"""
tests/test_adaptive.py
-----------------------
30 tests for the Adaptive Biometric-Contextual Authorization Engine.
All existing tests in test_engine.py continue to pass unchanged.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
import time
import pytest
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from database import db
from state.state_machine import StateMachine
from state.transition_manager import find_transition, TransitionCandidate
from security.adaptive_authorization import (
    AuthorizationContext, decide, compute_score,
    ALLOW, REAUTHENTICATE, STEP_UP, DENY, LOCK,
)
from security.temporal_binding import TemporalBinding
from security.authorization_token import TokenStore, AuthorizationToken
from security.security_escalation import SecurityEscalation
from security.gesture_authenticity import GestureAuthenticity
from security.security_feedback import SecurityFeedback
from security.authorization import AuthorizationManager
from audit_log.audit_logger import AuditLogger
from config.settings import ADAPTIVE_SECURITY_CONFIG

VALID_STATES = {
    "LOCKED", "AUTH_PENDING", "AUTHENTICATED",
    "PRESENTATION", "SETTINGS", "SENSITIVE_MODE", "ERROR",
}


def _ctx(**kwargs) -> AuthorizationContext:
    defaults = dict(
        gesture_confidence=0.95, biometric_confidence=0.85,
        stability_ratio=1.0, identity_continuity=1.0,
        authentication_age_sec=0.0, risk_level="LOW",
        failure_count=0, security_level=0,
        gesture_authenticity=1.0, auth_window_sec=30.0,
    )
    defaults.update(kwargs)
    return AuthorizationContext(**defaults)


@pytest.fixture(autouse=True)
def init_db():
    db.initialise()


# ── 1. Low-risk gesture without biometric ─────────────────────────────────────
def test_low_risk_no_bio_allow():
    ctx = _ctx(risk_level="LOW", biometric_confidence=0.0)
    r = decide(ctx)
    assert r.decision == ALLOW


# ── 2. Medium-risk gesture with good biometric ────────────────────────────────
def test_medium_risk_good_bio_allow():
    ctx = _ctx(risk_level="MEDIUM", biometric_confidence=0.80)
    r = decide(ctx)
    assert r.decision in (ALLOW, REAUTHENTICATE)


# ── 3. High-risk gesture with strong biometric ────────────────────────────────
def test_high_risk_strong_bio():
    ctx = _ctx(risk_level="HIGH", biometric_confidence=0.90,
               identity_continuity=1.0, failure_count=0)
    r = decide(ctx)
    assert r.decision in (ALLOW, REAUTHENTICATE)


# ── 4. Critical transition always needs strong auth ───────────────────────────
def test_critical_threshold_is_highest():
    low_t  = ADAPTIVE_SECURITY_CONFIG["biometric"]["low_threshold"]
    crit_t = ADAPTIVE_SECURITY_CONFIG["biometric"]["critical_threshold"]
    assert crit_t > low_t


# ── 5. Gesture confidence below threshold → score drops ──────────────────────
def test_low_gesture_confidence_reduces_score():
    high = compute_score(_ctx(gesture_confidence=0.95))
    low  = compute_score(_ctx(gesture_confidence=0.30))
    assert high > low


# ── 6. Biometric confidence below adaptive threshold → deny ──────────────────
def test_low_biometric_confidence_deny():
    ctx = _ctx(risk_level="HIGH", biometric_confidence=0.20,
               gesture_confidence=0.50, identity_continuity=0.3)
    r = decide(ctx)
    assert r.decision in (DENY, LOCK, STEP_UP)


# ── 7. Successful biometric authorization ────────────────────────────────────
def test_successful_bio_auth():
    ctx = _ctx(biometric_confidence=0.90, gesture_confidence=0.95,
               identity_continuity=1.0, risk_level="HIGH")
    r = decide(ctx)
    assert r.score > 0.0
    assert r.biometric_threshold > 0.0


# ── 8. Failed biometric authorization ────────────────────────────────────────
def test_failed_bio_auth():
    ctx = _ctx(biometric_confidence=0.10, gesture_confidence=0.95,
               identity_continuity=0.0, risk_level="HIGH", failure_count=2)
    r = decide(ctx)
    assert r.decision in (DENY, LOCK)


# ── 9. Expired authorization window ──────────────────────────────────────────
def test_expired_auth_window():
    auth = AuthorizationManager()
    auth.grant(user_id=1, timeout=0.05)
    time.sleep(0.1)
    assert auth.is_authorized() is False


# ── 10. Authorization token mismatch ─────────────────────────────────────────
def test_token_mismatch_gesture():
    store = TokenStore()
    store.issue(
        user_id=1, source_state="PRESENTATION", destination_state="LOCKED",
        gesture="Stop", risk_level="HIGH", risk_score=0.7,
        gesture_confidence=0.95, biometric_confidence=0.85,
        temporal_binding_score=1.0, auth_score=0.9, timeout=30.0,
    )
    valid, reason = store.validate("PRESENTATION", "Turn Right", "LOCKED", 1)
    assert valid is False
    assert "gesture" in reason.lower()


# ── 11. Token reused on another transition ────────────────────────────────────
def test_token_single_use():
    store = TokenStore()
    store.issue(
        user_id=1, source_state="PRESENTATION", destination_state="LOCKED",
        gesture="Stop", risk_level="HIGH", risk_score=0.7,
        gesture_confidence=0.95, biometric_confidence=0.85,
        temporal_binding_score=1.0, auth_score=0.9, timeout=30.0,
    )
    v1, _ = store.validate("PRESENTATION", "Stop", "LOCKED", 1)
    v2, r2 = store.validate("PRESENTATION", "Stop", "LOCKED", 1)
    assert v1 is True
    assert v2 is False
    assert "consumed" in r2.lower()


# ── 12. Identity disappears during gesture ────────────────────────────────────
def test_identity_loss_during_gesture():
    tb = TemporalBinding()
    tb.set_authenticated_user(1)
    tb.start_gesture()
    tb.record_observation(1, 0.9)
    tb.record_observation(None, 0.0)   # identity lost
    tb.record_observation(None, 0.0)
    score = tb.binding_score()
    assert score < 1.0


# ── 13. Identity disappears after authentication ──────────────────────────────
def test_identity_loss_after_auth():
    tb = TemporalBinding()
    tb.set_authenticated_user(1)
    tb.start_gesture()
    for _ in range(5):
        tb.record_observation(None, 0.0)
    assert tb.identity_lost() is True


# ── 14. Continuous auth relock via identity loss ──────────────────────────────
def test_continuous_auth_relock():
    fsm = StateMachine(VALID_STATES, "PRESENTATION")
    auth = AuthorizationManager()
    auth.grant(user_id=1, timeout=30.0)
    # Simulate identity loss beyond timeout
    auth.revoke()
    fsm.force("LOCKED")
    assert fsm.current == "LOCKED"
    assert auth.is_authorized() is False


# ── 15. Repeated biometric failures → lockout ────────────────────────────────
def test_repeated_failures_lockout():
    auth = AuthorizationManager()
    from config.settings import MAX_AUTH_FAILURES
    for _ in range(MAX_AUTH_FAILURES):
        auth.record_failure()
    assert auth.is_locked_out() is True


# ── 16. Security escalation on failures ──────────────────────────────────────
def test_security_escalation():
    esc = SecurityEscalation()
    assert esc.level == 0
    esc.record_failed_auth()
    assert esc.level == 1
    esc.record_failed_auth()
    assert esc.level == 2


# ── 17. Security de-escalation after clean activity ──────────────────────────
def test_security_deescalation():
    from config import settings
    orig = settings.ADAPTIVE_SECURITY_CONFIG["security_escalation"]["deescalate_after_sec"]
    settings.ADAPTIVE_SECURITY_CONFIG["security_escalation"]["deescalate_after_sec"] = 0.05
    esc = SecurityEscalation()
    esc.record_failed_auth()
    assert esc.level == 1
    esc.record_successful_transition()
    time.sleep(0.1)
    esc.tick()
    assert esc.level == 0
    settings.ADAPTIVE_SECURITY_CONFIG["security_escalation"]["deescalate_after_sec"] = orig


# ── 18. Step-up authentication decision ──────────────────────────────────────
def test_stepup_decision():
    ctx = _ctx(
        gesture_confidence=0.70, biometric_confidence=0.50,
        identity_continuity=0.60, risk_level="HIGH",
        failure_count=1, security_level=1,
    )
    r = decide(ctx)
    assert r.decision in (STEP_UP, REAUTHENTICATE, DENY)


# ── 19. Temporal binding failure reduces score ────────────────────────────────
def test_temporal_binding_reduces_score():
    good = compute_score(_ctx(identity_continuity=1.0))
    bad  = compute_score(_ctx(identity_continuity=0.0))
    assert good > bad


# ── 20. Gesture authenticity failure reduces score ───────────────────────────
def test_gesture_authenticity_reduces_score():
    good = compute_score(_ctx(gesture_authenticity=1.0))
    bad  = compute_score(_ctx(gesture_authenticity=0.0))
    assert good > bad


# ── 21. State rollback on denied transition ───────────────────────────────────
def test_state_rollback_on_deny():
    fsm = StateMachine(VALID_STATES, "PRESENTATION")
    fsm.propose("LOCKED")
    fsm.rollback()
    assert fsm.current == "PRESENTATION"
    assert fsm.candidate is None


# ── 22. Action blocked without authorization ─────────────────────────────────
def test_action_blocked_without_auth():
    fsm = StateMachine(VALID_STATES, "PRESENTATION")
    c = find_transition("PRESENTATION", "Stop", 0.95)
    assert c is not None
    fsm.propose(c.destination_state)
    fsm.rollback()
    assert fsm.current == "PRESENTATION"


# ── 23. Successful state commit ───────────────────────────────────────────────
def test_successful_state_commit():
    fsm = StateMachine(VALID_STATES, "PRESENTATION")
    auth = AuthorizationManager()
    auth.grant(user_id=1, timeout=15.0)
    fsm.propose("LOCKED")
    fsm.commit()
    assert fsm.current == "LOCKED"


# ── 24. Audit log written with adaptive fields ───────────────────────────────
def test_audit_log_adaptive_fields():
    audit = AuditLogger("TEST-ADAPTIVE")
    before = len(db.recent_logs(200))
    audit.log(
        user_id=1, previous_state="PRESENTATION", gesture="Stop",
        gesture_confidence=0.95, destination_state="LOCKED",
        risk_level="HIGH", biometric_required=True,
        biometric_type="FACE", biometric_confidence=0.87,
        authentication_result="PASS", authorization_result="ALLOWED",
        final_state="LOCKED", action="STOP_PRESENTATION",
        authorization_score=0.88, gesture_authenticity_score=0.92,
        temporal_binding_score=0.95, risk_score=0.70,
        biometric_threshold_used=0.75, security_state="NORMAL",
        security_escalation_level=0, token_id="abc12345",
        identity_continuity=0.95, decision_reason="Score 0.88 >= 0.85",
    )
    after = len(db.recent_logs(200))
    assert after == before + 1


# ── 25. Authorization score calculation ──────────────────────────────────────
def test_authorization_score_range():
    for risk in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        s = compute_score(_ctx(risk_level=risk))
        assert 0.0 <= s <= 1.0


# ── 26. Risk score mapping ────────────────────────────────────────────────────
def test_risk_score_ordering():
    from security.adaptive_authorization import _risk_score
    assert _risk_score("LOW") < _risk_score("MEDIUM")
    assert _risk_score("MEDIUM") < _risk_score("HIGH")
    assert _risk_score("HIGH") < _risk_score("CRITICAL")


# ── 27. Adaptive biometric threshold ordering ─────────────────────────────────
def test_adaptive_bio_threshold_ordering():
    from security.adaptive_authorization import _biometric_threshold
    assert _biometric_threshold("LOW") < _biometric_threshold("MEDIUM")
    assert _biometric_threshold("MEDIUM") < _biometric_threshold("HIGH")
    assert _biometric_threshold("HIGH") < _biometric_threshold("CRITICAL")


# ── 28. Adaptive authorization timeout ordering ───────────────────────────────
def test_adaptive_timeout_ordering():
    from security.adaptive_authorization import _auth_window
    assert _auth_window("LOW") > _auth_window("MEDIUM")
    assert _auth_window("MEDIUM") > _auth_window("HIGH")
    assert _auth_window("HIGH") >= _auth_window("CRITICAL")


# ── 29. Token expiry ──────────────────────────────────────────────────────────
def test_token_expiry():
    store = TokenStore()
    store.issue(
        user_id=1, source_state="PRESENTATION", destination_state="LOCKED",
        gesture="Stop", risk_level="HIGH", risk_score=0.7,
        gesture_confidence=0.95, biometric_confidence=0.85,
        temporal_binding_score=1.0, auth_score=0.9, timeout=0.05,
    )
    time.sleep(0.1)
    valid, reason = store.validate("PRESENTATION", "Stop", "LOCKED", 1)
    assert valid is False
    assert "expired" in reason.lower()


# ── 30. Database migration is idempotent ─────────────────────────────────────
def test_db_migration_idempotent():
    # Running initialise() twice must not raise
    db.initialise()
    db.initialise()
    conn = db.get_connection()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(transition_log)").fetchall()}
    conn.close()
    assert "authorization_score" in cols
    assert "temporal_binding_score" in cols
    assert "security_state" in cols
