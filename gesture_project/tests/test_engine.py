"""
tests/test_engine.py
--------------------
Automated tests for the Biometric-Gated State Transition Engine.
Run with:  python -m pytest tests/test_engine.py -v
"""

import sys
import os
import time
import pytest
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from state.state_machine import StateMachine
from state.transition_manager import find_transition, TransitionCandidate
from security.policy_engine import biometric_required
from security.authorization import AuthorizationManager
from config.settings import BIOMETRIC_CONFIDENCE_THRESHOLD


# ── Fixtures ───────────────────────────────────────────────────────────────────

VALID_STATES = {
    "LOCKED", "AUTH_PENDING", "AUTHENTICATED",
    "PRESENTATION", "SETTINGS", "SENSITIVE_MODE", "ERROR",
}


@pytest.fixture
def fsm():
    return StateMachine(VALID_STATES, initial="LOCKED")


@pytest.fixture
def auth():
    return AuthorizationManager()


# ══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStateMachine:

    def test_initial_state(self, fsm):
        assert fsm.current == "LOCKED"

    def test_valid_transition(self, fsm):
        assert fsm.propose("AUTH_PENDING") is True
        fsm.commit()
        assert fsm.current == "AUTH_PENDING"
        assert fsm.previous == "LOCKED"

    def test_invalid_destination_rejected(self, fsm):
        assert fsm.propose("NONEXISTENT_STATE") is False
        assert fsm.current == "LOCKED"

    def test_rollback_keeps_state(self, fsm):
        fsm.propose("AUTH_PENDING")
        fsm.rollback()
        assert fsm.current == "LOCKED"
        assert fsm.candidate is None

    def test_force_state(self, fsm):
        fsm.force("ERROR")
        assert fsm.current == "ERROR"

    def test_commit_without_propose_raises(self, fsm):
        with pytest.raises(RuntimeError):
            fsm.commit()

    def test_state_sequence(self, fsm):
        for dest in ("AUTH_PENDING", "AUTHENTICATED", "PRESENTATION"):
            fsm.propose(dest)
            fsm.commit()
        assert fsm.current == "PRESENTATION"
        assert fsm.previous == "AUTHENTICATED"


# ══════════════════════════════════════════════════════════════════════════════
# TRANSITION MANAGER TESTS (requires DB)
# ══════════════════════════════════════════════════════════════════════════════

class TestTransitionManager:

    @pytest.fixture(autouse=True)
    def _init_db(self):
        from database import db
        db.initialise()

    def test_valid_transition_found(self):
        c = find_transition("PRESENTATION", "Turn Right", 0.90)
        assert c is not None
        assert c.destination_state == "PRESENTATION"
        assert c.action == "NEXT_SLIDE"
        assert c.risk_level == "LOW"

    def test_no_transition_for_unknown_gesture(self):
        c = find_transition("PRESENTATION", "UNKNOWN_GESTURE", 0.90)
        assert c is None

    def test_no_transition_for_wrong_state(self):
        c = find_transition("LOCKED", "Turn Right", 0.90)
        assert c is None

    def test_low_confidence_rejected(self):
        # confidence below the row threshold (0.80)
        c = find_transition("PRESENTATION", "Turn Right", 0.50)
        assert c is None

    def test_high_risk_transition_has_bio_required(self):
        c = find_transition("PRESENTATION", "Stop", 0.95)
        assert c is not None
        assert c.risk_level == "HIGH"
        assert c.biometric_required is True

    def test_locked_start_transition(self):
        c = find_transition("LOCKED", "Start", 0.85)
        assert c is not None
        assert c.destination_state == "AUTH_PENDING"


# ══════════════════════════════════════════════════════════════════════════════
# POLICY ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_candidate(risk: str, bio_req: bool) -> TransitionCandidate:
    return TransitionCandidate(
        source_state="PRESENTATION", gesture="Stop",
        destination_state="LOCKED", risk_level=risk,
        biometric_required=bio_req, biometric_type="FACE",
        confidence_threshold=0.80, authorization_timeout=15.0,
        action="STOP_PRESENTATION",
    )


class TestPolicyEngine:

    def test_low_risk_never_requires_bio(self):
        assert biometric_required(_make_candidate("LOW", True)) is False

    def test_high_risk_always_requires_bio(self):
        assert biometric_required(_make_candidate("HIGH", False)) is True

    def test_critical_always_requires_bio(self):
        assert biometric_required(_make_candidate("CRITICAL", False)) is True

    def test_medium_follows_transition_flag_true(self):
        assert biometric_required(_make_candidate("MEDIUM", True)) is True

    def test_medium_follows_transition_flag_false(self):
        assert biometric_required(_make_candidate("MEDIUM", False)) is False


# ══════════════════════════════════════════════════════════════════════════════
# AUTHORIZATION MANAGER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthorizationManager:

    def test_not_authorized_initially(self, auth):
        assert auth.is_authorized() is False

    def test_grant_authorizes(self, auth):
        auth.grant(user_id=1, timeout=10.0)
        assert auth.is_authorized() is True
        assert auth.authorized_user_id == 1

    def test_authorization_expires(self, auth):
        auth.grant(user_id=1, timeout=0.05)
        time.sleep(0.1)
        assert auth.is_authorized() is False

    def test_revoke_clears_auth(self, auth):
        auth.grant(user_id=1, timeout=10.0)
        auth.revoke()
        assert auth.is_authorized() is False

    def test_remaining_seconds_decreases(self, auth):
        auth.grant(user_id=1, timeout=5.0)
        r1 = auth.remaining_seconds()
        time.sleep(0.1)
        r2 = auth.remaining_seconds()
        assert r2 < r1

    def test_failure_counter(self, auth):
        auth.record_failure()
        auth.record_failure()
        assert auth.failure_count == 2

    def test_lockout_after_max_failures(self, auth):
        from config.settings import MAX_AUTH_FAILURES
        for _ in range(MAX_AUTH_FAILURES):
            auth.record_failure()
        assert auth.is_locked_out() is True

    def test_lockout_expires(self, auth):
        # Patch lockout duration to near-zero for test speed
        from config import settings
        original = settings.AUTH_LOCKOUT_SEC
        settings.AUTH_LOCKOUT_SEC = 0.05
        from config.settings import MAX_AUTH_FAILURES
        for _ in range(MAX_AUTH_FAILURES):
            auth.record_failure()
        time.sleep(0.15)
        assert auth.is_locked_out() is False
        settings.AUTH_LOCKOUT_SEC = original

    def test_reset_failures(self, auth):
        auth.record_failure()
        auth.reset_failures()
        assert auth.failure_count == 0


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY / INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityIntegration:

    @pytest.fixture(autouse=True)
    def _init_db(self):
        from database import db
        db.initialise()

    def test_unauthorized_gesture_in_locked_state(self):
        """Turn Right in LOCKED state has no transition — must be denied."""
        c = find_transition("LOCKED", "Turn Right", 0.95)
        assert c is None

    def test_protected_transition_requires_bio(self):
        """PRESENTATION + Stop must require biometric."""
        c = find_transition("PRESENTATION", "Stop", 0.95)
        assert c is not None
        assert biometric_required(c) is True

    def test_low_risk_transition_no_bio(self):
        """PRESENTATION + Turn Right must NOT require biometric."""
        c = find_transition("PRESENTATION", "Turn Right", 0.90)
        assert c is not None
        assert biometric_required(c) is False

    def test_action_not_executed_without_auth(self, fsm):
        """State must not change if biometric is required but not provided."""
        fsm.force("PRESENTATION")
        c = find_transition("PRESENTATION", "Stop", 0.95)
        assert c is not None
        # Propose but do NOT commit (simulating failed auth)
        fsm.propose(c.destination_state)
        fsm.rollback()
        assert fsm.current == "PRESENTATION"

    def test_full_authorized_flow(self, fsm, auth):
        """Full happy path: gesture → bio → auth → state change → action."""
        fsm.force("PRESENTATION")
        c = find_transition("PRESENTATION", "Stop", 0.95)
        assert c is not None
        assert biometric_required(c) is True

        # Simulate successful biometric
        auth.grant(user_id=1, timeout=15.0)
        assert auth.is_authorized() is True

        fsm.propose(c.destination_state)
        fsm.commit()
        assert fsm.current == "LOCKED"

    def test_expired_auth_window_blocks_transition(self, fsm, auth):
        """Expired authorization must not allow protected transition."""
        fsm.force("PRESENTATION")
        auth.grant(user_id=1, timeout=0.05)
        time.sleep(0.1)
        assert auth.is_authorized() is False

    def test_repeated_failures_trigger_lockout(self, auth):
        from config.settings import MAX_AUTH_FAILURES
        for _ in range(MAX_AUTH_FAILURES):
            auth.record_failure()
        assert auth.is_locked_out() is True

    def test_audit_log_written(self):
        from database import db
        from audit_log.audit_logger import AuditLogger
        audit = AuditLogger("TEST-SESSION")
        before = len(db.recent_logs(100))
        audit.log(
            user_id=None,
            previous_state="PRESENTATION",
            gesture="Stop",
            gesture_confidence=0.95,
            destination_state="LOCKED",
            risk_level="HIGH",
            biometric_required=True,
            biometric_type="FACE",
            biometric_confidence=0.45,
            authentication_result="FAIL",
            authorization_result="DENIED",
            final_state="PRESENTATION",
            action="NONE",
            reason="Test entry",
        )
        after = len(db.recent_logs(100))
        assert after == before + 1
