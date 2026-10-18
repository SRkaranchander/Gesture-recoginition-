"""
main.py
-------
Adaptive Biometric-Contextual Authorization Engine.
Wraps the original gesture pipeline and routes through the adaptive engine
when ADAPTIVE_SECURITY=True, or the original fixed-threshold path otherwise.
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys
import time
import argparse
import collections
import logging
import threading
from typing import Optional

if not hasattr(mp, "solutions"):
    sys.exit(
        "\n[ERROR] Unsupported MediaPipe version in this Python environment.\n"
        "Please run this project using its dedicated virtual environment:\n"
        "  ..\\.venv\\Scripts\\python.exe main.py\n"
        "or activate the virtual environment first:\n"
        "  ..\\.venv\\Scripts\\Activate.ps1\n"
    )

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.gesture_config import (
    GESTURE_LABELS, SEQUENCE_LENGTH,
    full_feature_vector, draw_landmark_skeleton, COLOURS,
)
from config.settings import (
    GESTURE_CONFIDENCE_THRESHOLD, STABILITY_FRAMES,
    GESTURE_COOLDOWN_SEC, AUTHORIZATION_TIMEOUT_SEC,
    CONTINUOUS_AUTH, CONTINUOUS_AUTH_INTERVAL_SEC,
    CONTINUOUS_AUTH_LOSS_TIMEOUT_SEC, DEFAULT_MODEL,
    DASHBOARD_WIDTH, DASHBOARD_HEIGHT, SECURITY_EVENT_HISTORY,
    BIOMETRIC_CONFIDENCE_THRESHOLD, ADAPTIVE_SECURITY,
    ADAPTIVE_SECURITY_CONFIG,
)
from database import db
from state.state_machine import StateMachine
from state.transition_manager import find_transition, TransitionCandidate
from security.policy_engine import biometric_required as policy_bio_required
from security.authorization import AuthorizationManager
from security.session_manager import SessionManager
from security.adaptive_authorization import decide, AuthorizationContext
from security.adaptive_authorization import ALLOW, REAUTHENTICATE, STEP_UP, DENY, LOCK
from security.temporal_binding import TemporalBinding
from security.authorization_token import TokenStore
from security.security_escalation import SecurityEscalation
from security.gesture_authenticity import GestureAuthenticity
from security.security_feedback import SecurityFeedback
from biometric.face_auth import verify_frame, draw_face_overlay, authenticate
from actions.system_actions import execute as execute_action
from audit_log.audit_logger import AuditLogger
from ui.overlays import (
    draw_state_panel, draw_gesture_panel, draw_auth_panel,
    draw_transition_panel, draw_security_events,
    draw_fps_and_session, draw_state_graph,
    draw_biometric_request_overlay,
    draw_adaptive_panel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


def load_model(model_name: str):
    path = os.path.join(ROOT, "models", f"{model_name}_model.pkl")
    if not os.path.exists(path):
        logger.error("Model not found: %s", path)
        sys.exit(1)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Loaded model: %s", path)
    return bundle["model"], bundle["scaler"], bundle["feat"]


def predict(model, scaler, feat_type: str, sequence: list) -> tuple[Optional[str], float]:
    if len(sequence) < SEQUENCE_LENGTH:
        return None, 0.0
    arr = np.array(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)
    X = arr.flatten().reshape(1, -1) if feat_type == "full" \
        else np.concatenate([arr.mean(axis=0), arr.std(axis=0)]).reshape(1, -1)
    X_s = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_s)[0]
        gid = int(np.argmax(proba))
        return GESTURE_LABELS.get(gid, "Unknown"), float(proba[gid])
    gid = int(model.predict(X_s)[0])
    return GESTURE_LABELS.get(gid, "Unknown"), 1.0


class BiometricGate:
    """Non-blocking background face authentication thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._result: Optional[tuple[Optional[int], float]] = None
        self._deadline: float = 0.0

    def start(self, cap: cv2.VideoCapture, timeout: float) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._result = None
            self._deadline = time.time() + timeout
        threading.Thread(target=self._run, args=(cap, timeout), daemon=True).start()

    def _run(self, cap: cv2.VideoCapture, timeout: float) -> None:
        uid, conf = authenticate(cap, timeout)
        with self._lock:
            self._result = (uid, conf)
            self._running = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def result(self) -> Optional[tuple[Optional[int], float]]:
        with self._lock:
            return self._result

    @property
    def countdown(self) -> float:
        return max(0.0, self._deadline - time.time())

    def reset(self) -> None:
        with self._lock:
            self._running = False
            self._result = None


def _adaptive_authorize(
    candidate: TransitionCandidate,
    gesture_conf: float,
    bio_conf: float,
    stable_count: int,
    auth_mgr: AuthorizationManager,
    temporal: TemporalBinding,
    authenticity: GestureAuthenticity,
    escalation: SecurityEscalation,
    auth_time: Optional[float],
) -> tuple:
    """
    Build AuthorizationContext, run adaptive scoring, return
    (result, ctx) where result is an AuthorizationResult dataclass.
    """
    risk_level = candidate.risk_level
    window = ADAPTIVE_SECURITY_CONFIG["authorization_window"].get(risk_level, 15.0)
    age = (time.time() - auth_time) if auth_time else window

    ctx = AuthorizationContext(
        gesture_confidence=gesture_conf,
        biometric_confidence=bio_conf,
        stability_ratio=min(1.0, stable_count / STABILITY_FRAMES),
        identity_continuity=temporal.binding_score(),
        authentication_age_sec=age,
        risk_level=risk_level,
        failure_count=auth_mgr.failure_count,
        security_level=escalation.level,
        gesture_authenticity=authenticity.score(),
        auth_window_sec=window,
    )
    return decide(ctx), ctx


def main(model_name: str = DEFAULT_MODEL) -> None:
    db.initialise()
    session  = SessionManager()
    auth_mgr = AuthorizationManager()
    audit    = AuditLogger(session.session_id)
    bio_gate = BiometricGate()

    # Adaptive subsystems
    temporal    = TemporalBinding()
    token_store = TokenStore()
    escalation  = SecurityEscalation()
    authenticity = GestureAuthenticity()
    feedback    = SecurityFeedback(escalation)

    conn = db.get_connection()
    valid_states = {r["state_name"] for r in
                    conn.execute("SELECT state_name FROM states").fetchall()}
    conn.close()
    fsm = StateMachine(valid_states, initial="LOCKED")
    all_transitions = db.get_all_transitions()

    model, scaler, feat_type = load_model(model_name)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.75, min_tracking_confidence=0.65,
    )

    cv2.namedWindow("Adaptive Biometric-Gated Engine", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Adaptive Biometric-Gated Engine", DASHBOARD_WIDTH, DASHBOARD_HEIGHT)

    # State variables
    sequence: list = []
    last_gesture: Optional[str] = None
    stable_count: int = 0
    last_fired_gesture: Optional[str] = None
    last_fired_time: float = 0.0
    prev_time: float = time.time()
    last_cont_check: float = 0.0
    identity_lost_at: Optional[float] = None
    last_auth_time: Optional[float] = None

    security_events: collections.deque = collections.deque(maxlen=SECURITY_EVENT_HISTORY)
    last_candidate: Optional[TransitionCandidate] = None
    last_decision: str = "—"
    last_auth_result: Optional[object] = None   # AuthorizationResult
    bio_conf_display: float = 0.0
    pending_candidate: Optional[TransitionCandidate] = None

    def add_event(msg: str, colour=(200, 200, 200)) -> None:
        security_events.appendleft((time.strftime("%H:%M:%S"), msg, colour))

    add_event("System started — LOCKED", (0, 200, 80))
    logger.info("Session: %s  adaptive=%s", session.session_id, ADAPTIVE_SECURITY)

    # ── Frame loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-8)
        prev_time = now

        gesture: Optional[str] = None
        conf: float = 0.0

        escalation.tick()

        # ── Gesture recognition (existing pipeline unchanged) ──────────────────
        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            fv = full_feature_vector(hl)
            sequence.append(fv)
            if len(sequence) > SEQUENCE_LENGTH * 2:
                sequence = sequence[-SEQUENCE_LENGTH:]
            gesture, conf = predict(model, scaler, feat_type, sequence)
            if gesture == last_gesture:
                stable_count += 1
            else:
                stable_count = 0
                last_gesture = gesture
                authenticity.start()
            authenticity.add_frame(fv)
            draw_landmark_skeleton(frame, hl, mp_hands, mp_drawing, gesture or "default")
        else:
            sequence = []
            stable_count = 0

        # ── Continuous authentication + temporal binding ───────────────────────
        uid_live, conf_live = verify_frame(frame)
        temporal.record_observation(uid_live, conf_live)
        draw_face_overlay(frame, uid_live, conf_live)
        if uid_live is not None:
            bio_conf_display = conf_live

        if CONTINUOUS_AUTH and fsm.current not in ("LOCKED", "AUTH_PENDING"):
            if now - last_cont_check >= CONTINUOUS_AUTH_INTERVAL_SEC:
                last_cont_check = now
                if uid_live is None:
                    if identity_lost_at is None:
                        identity_lost_at = now
                        add_event("Identity lost — monitoring", (0, 140, 255))
                        feedback.on_identity_loss()
                    elif now - identity_lost_at > CONTINUOUS_AUTH_LOSS_TIMEOUT_SEC:
                        add_event("AUTO-RELOCK: identity lost", (0, 60, 220))
                        auth_mgr.revoke()
                        token_store.invalidate("identity lost")
                        fsm.force("LOCKED")
                        identity_lost_at = None
                else:
                    identity_lost_at = None

        # ── Biometric gate result processing ──────────────────────────────────
        if pending_candidate is not None and not bio_gate.is_running:
            bio_result = bio_gate.result
            if bio_result is not None:
                uid_bio, conf_bio = bio_result
                bio_conf_display = conf_bio
                adaptive_threshold = ADAPTIVE_SECURITY_CONFIG["biometric"].get(
                    f"{pending_candidate.risk_level.lower()}_threshold",
                    BIOMETRIC_CONFIDENCE_THRESHOLD,
                ) if ADAPTIVE_SECURITY else BIOMETRIC_CONFIDENCE_THRESHOLD

                auth_ok = uid_bio is not None and conf_bio >= adaptive_threshold

                if auth_ok:
                    last_auth_time = now
                    temporal.set_authenticated_user(uid_bio)

                    if ADAPTIVE_SECURITY:
                        ar, ctx = _adaptive_authorize(
                            pending_candidate, conf, conf_bio,
                            stable_count, auth_mgr, temporal,
                            authenticity, escalation, last_auth_time,
                        )
                        last_auth_result = ar
                        authorized = ar.decision == ALLOW
                    else:
                        authorized = True
                        ar = None

                    if authorized:
                        timeout = ar.authorization_timeout if ar else pending_candidate.authorization_timeout
                        auth_mgr.grant(uid_bio, timeout)
                        auth_mgr.reset_failures()
                        token = token_store.issue(
                            user_id=uid_bio,
                            source_state=pending_candidate.source_state,
                            destination_state=pending_candidate.destination_state,
                            gesture=pending_candidate.gesture,
                            risk_level=pending_candidate.risk_level,
                            risk_score=ar.risk_score if ar else 0.5,
                            gesture_confidence=conf,
                            biometric_confidence=conf_bio,
                            temporal_binding_score=temporal.binding_score(),
                            auth_score=ar.score if ar else 1.0,
                            timeout=timeout,
                            security_level=escalation.level,
                        )
                        valid, reason = token_store.validate(
                            pending_candidate.source_state,
                            pending_candidate.gesture,
                            pending_candidate.destination_state,
                            uid_bio,
                        )
                        if valid:
                            dest_state = "AUTHENTICATED" if pending_candidate.destination_state == "AUTH_PENDING" else pending_candidate.destination_state
                            fsm.force(dest_state)
                            execute_action(pending_candidate.action)
                            feedback.on_allow(pending_candidate.gesture)
                            feedback.on_successful_auth()
                            add_event(f"AUTH OK -> {dest_state}", (0, 200, 80))
                            last_decision = "ALLOWED"
                            audit.log(
                                user_id=uid_bio,
                                previous_state=pending_candidate.source_state,
                                gesture=pending_candidate.gesture,
                                gesture_confidence=conf,
                                destination_state=pending_candidate.destination_state,
                                risk_level=pending_candidate.risk_level,
                                biometric_required=True,
                                biometric_type=pending_candidate.biometric_type,
                                biometric_confidence=conf_bio,
                                authentication_result="PASS",
                                authorization_result="ALLOWED",
                                final_state=fsm.current,
                                action=pending_candidate.action,
                                authorization_score=ar.score if ar else None,
                                temporal_binding_score=temporal.binding_score(),
                                risk_score=ar.risk_score if ar else None,
                                biometric_threshold_used=adaptive_threshold,
                                security_state=escalation.level_name,
                                security_escalation_level=escalation.level,
                                token_id=token.token_id,
                                identity_continuity=temporal.binding_score(),
                                decision_reason=ar.reason if ar else "PASS",
                            )
                        else:
                            fsm.rollback()
                            last_decision = f"TOKEN INVALID: {reason}"
                            add_event(f"Token invalid: {reason}", (0, 60, 220))
                    else:
                        fsm.rollback()
                        last_decision = f"ADAPTIVE DENY: {ar.decision}"
                        add_event(f"Adaptive: {ar.decision}", (0, 60, 220))
                        audit.log(
                            user_id=uid_bio,
                            previous_state=pending_candidate.source_state,
                            gesture=pending_candidate.gesture,
                            gesture_confidence=conf,
                            destination_state=pending_candidate.destination_state,
                            risk_level=pending_candidate.risk_level,
                            biometric_required=True,
                            biometric_type=pending_candidate.biometric_type,
                            biometric_confidence=conf_bio,
                            authentication_result="PASS",
                            authorization_result=ar.decision,
                            final_state=fsm.current,
                            action="NONE",
                            authorization_score=ar.score,
                            decision_reason=ar.reason,
                            security_state=escalation.level_name,
                        )
                else:
                    failures = auth_mgr.record_failure()
                    feedback.on_failed_biometric(conf_bio)
                    add_event(f"BIO FAIL ({conf_bio*100:.0f}%)", (0, 60, 220))
                    if auth_mgr.is_locked_out():
                        fsm.force("LOCKED")
                        feedback.on_lockout()
                        add_event("LOCKOUT — too many failures", (0, 60, 220))
                    last_decision = "DENIED"
                    audit.log(
                        user_id=None,
                        previous_state=pending_candidate.source_state,
                        gesture=pending_candidate.gesture,
                        gesture_confidence=conf,
                        destination_state=pending_candidate.destination_state,
                        risk_level=pending_candidate.risk_level,
                        biometric_required=True,
                        biometric_type=pending_candidate.biometric_type,
                        biometric_confidence=conf_bio,
                        authentication_result="FAIL",
                        authorization_result="DENIED",
                        final_state=fsm.current,
                        action="NONE",
                        biometric_threshold_used=adaptive_threshold,
                        security_state=escalation.level_name,
                        security_escalation_level=escalation.level,
                        reason=f"conf {conf_bio:.3f} < threshold {adaptive_threshold:.3f}",
                    )
                pending_candidate = None
                bio_gate.reset()

        # ── Gesture event → Transition Engine ─────────────────────────────────
        gesture_fired = (
            stable_count == STABILITY_FRAMES
            and gesture is not None
            and conf >= GESTURE_CONFIDENCE_THRESHOLD
            and pending_candidate is None
            and not bio_gate.is_running
        )
        if gesture_fired:
            if gesture == last_fired_gesture and (now - last_fired_time) < GESTURE_COOLDOWN_SEC:
                gesture_fired = False

        if gesture_fired:
            last_fired_gesture = gesture
            last_fired_time = now
            temporal.start_gesture()
            add_event(f"Gesture: {gesture} ({conf*100:.0f}%)", COLOURS.get(gesture, (200, 200, 200)))

            candidate = find_transition(fsm.current, gesture, conf)

            if candidate is None:
                add_event(f"No transition: {fsm.current}+{gesture}", (100, 100, 120))
                feedback.on_deny("no transition")
                last_decision = "NO TRANSITION"
                audit.log(
                    user_id=auth_mgr.authorized_user_id,
                    previous_state=fsm.current, gesture=gesture,
                    gesture_confidence=conf, destination_state=fsm.current,
                    risk_level="NONE", biometric_required=False,
                    biometric_type=None, biometric_confidence=None,
                    authentication_result=None, authorization_result="DENIED",
                    final_state=fsm.current, action="NONE",
                    reason="No valid transition found",
                    security_state=escalation.level_name,
                )
            else:
                last_candidate = candidate
                needs_bio = policy_bio_required(candidate)

                if candidate.source_state == "LOCKED" or candidate.action == "REQUEST_AUTH":
                    if uid_live is not None and conf_live >= BIOMETRIC_CONFIDENCE_THRESHOLD:
                        auth_mgr.grant(uid_live, AUTHORIZATION_TIMEOUT_SEC)
                        auth_mgr.reset_failures()
                        last_auth_time = now
                        temporal.set_authenticated_user(uid_live)
                        fsm.force("AUTHENTICATED")
                        feedback.on_allow(gesture)
                        feedback.on_successful_auth()
                        add_event(f"Face Verified ({conf_live*100:.0f}%) -> AUTHENTICATED", (0, 200, 80))
                        last_decision = "AUTHENTICATED"
                        audit.log(
                            user_id=uid_live,
                            previous_state="LOCKED", gesture=gesture,
                            gesture_confidence=conf,
                            destination_state="AUTHENTICATED",
                            risk_level="MEDIUM", biometric_required=True,
                            biometric_type="FACE", biometric_confidence=conf_live,
                            authentication_result="PASS", authorization_result="ALLOWED",
                            final_state="AUTHENTICATED", action="AUTHENTICATE",
                            security_state=escalation.level_name,
                            decision_reason="Live face biometric match",
                        )
                    else:
                        fsm.force("AUTH_PENDING")
                        pending_candidate = candidate
                        bio_gate.start(cap, candidate.authorization_timeout)
                        add_event("BIO REQUIRED: Face verify", (0, 140, 255))
                        last_decision = "BIO PENDING"
                elif ADAPTIVE_SECURITY:
                    bio_c = bio_conf_display if needs_bio else 0.0
                    ar, ctx = _adaptive_authorize(
                        candidate, conf, bio_c, stable_count,
                        auth_mgr, temporal, authenticity, escalation, last_auth_time,
                    )
                    last_auth_result = ar

                    if not needs_bio and ar.decision == ALLOW:
                        fsm.commit()
                        execute_action(candidate.action)
                        feedback.on_allow(gesture)
                        add_event(f"{candidate.action} [{ar.score:.2f}]", (0, 200, 80))
                        last_decision = f"ALLOW ({ar.score:.2f})"
                        audit.log(
                            user_id=auth_mgr.authorized_user_id,
                            previous_state=candidate.source_state, gesture=gesture,
                            gesture_confidence=conf,
                            destination_state=candidate.destination_state,
                            risk_level=candidate.risk_level, biometric_required=False,
                            biometric_type=None, biometric_confidence=None,
                            authentication_result=None, authorization_result="ALLOWED",
                            final_state=fsm.current, action=candidate.action,
                            authorization_score=ar.score, risk_score=ar.risk_score,
                            gesture_authenticity_score=authenticity.score(),
                            temporal_binding_score=temporal.binding_score(),
                            security_state=escalation.level_name,
                            security_escalation_level=escalation.level,
                            decision_reason=ar.reason,
                        )
                    elif needs_bio and auth_mgr.is_authorized() and ar.decision == ALLOW:
                        fsm.commit()
                        execute_action(candidate.action)
                        feedback.on_allow(gesture)
                        add_event(f"WINDOW+ADAPTIVE: {candidate.action}", (0, 200, 80))
                        last_decision = f"ALLOW-WINDOW ({ar.score:.2f})"
                        audit.log(
                            user_id=auth_mgr.authorized_user_id,
                            previous_state=candidate.source_state, gesture=gesture,
                            gesture_confidence=conf,
                            destination_state=candidate.destination_state,
                            risk_level=candidate.risk_level, biometric_required=True,
                            biometric_type=candidate.biometric_type,
                            biometric_confidence=bio_conf_display,
                            authentication_result="WINDOW",
                            authorization_result="ALLOWED",
                            final_state=fsm.current, action=candidate.action,
                            authorization_score=ar.score, risk_score=ar.risk_score,
                            security_state=escalation.level_name,
                            decision_reason=ar.reason,
                        )
                    elif ar.decision in (DENY, LOCK):
                        fsm.rollback()
                        if ar.decision == LOCK:
                            fsm.force("LOCKED")
                            feedback.on_lockout()
                        feedback.on_deny(ar.reason)
                        add_event(f"ADAPTIVE {ar.decision}: {ar.reason[:25]}", (0, 60, 220))
                        last_decision = ar.decision
                        audit.log(
                            user_id=None,
                            previous_state=candidate.source_state, gesture=gesture,
                            gesture_confidence=conf,
                            destination_state=candidate.destination_state,
                            risk_level=candidate.risk_level, biometric_required=needs_bio,
                            biometric_type=candidate.biometric_type,
                            biometric_confidence=None, authentication_result=None,
                            authorization_result=ar.decision,
                            final_state=fsm.current, action="NONE",
                            authorization_score=ar.score, risk_score=ar.risk_score,
                            security_state=escalation.level_name,
                            decision_reason=ar.reason,
                        )
                    else:
                        # REAUTHENTICATE or STEP_UP — trigger biometric
                        fsm.rollback()
                        pending_candidate = candidate
                        bio_gate.start(cap, candidate.authorization_timeout)
                        add_event(f"BIO REQUIRED ({ar.decision}): {gesture}", (0, 140, 255))
                        last_decision = f"BIO PENDING ({ar.decision})"
                else:
                    # ── Original fixed-threshold path ──────────────────────────
                    if not needs_bio:
                        fsm.commit()
                        execute_action(candidate.action)
                        feedback.on_allow(gesture)
                        add_event(f"{candidate.action} -> {candidate.destination_state}", (0, 200, 80))
                        last_decision = "ALLOWED"
                        audit.log(
                            user_id=auth_mgr.authorized_user_id,
                            previous_state=candidate.source_state, gesture=gesture,
                            gesture_confidence=conf,
                            destination_state=candidate.destination_state,
                            risk_level=candidate.risk_level, biometric_required=False,
                            biometric_type=None, biometric_confidence=None,
                            authentication_result=None, authorization_result="ALLOWED",
                            final_state=fsm.current, action=candidate.action,
                        )
                    elif auth_mgr.is_authorized():
                        fsm.commit()
                        execute_action(candidate.action)
                        feedback.on_allow(gesture)
                        add_event(f"AUTH WINDOW: {candidate.action}", (0, 200, 80))
                        last_decision = "ALLOWED (window)"
                        audit.log(
                            user_id=auth_mgr.authorized_user_id,
                            previous_state=candidate.source_state, gesture=gesture,
                            gesture_confidence=conf,
                            destination_state=candidate.destination_state,
                            risk_level=candidate.risk_level, biometric_required=True,
                            biometric_type=candidate.biometric_type,
                            biometric_confidence=None,
                            authentication_result="WINDOW",
                            authorization_result="ALLOWED",
                            final_state=fsm.current, action=candidate.action,
                        )
                    elif auth_mgr.is_locked_out():
                        fsm.rollback()
                        add_event("LOCKED OUT — wait", (0, 60, 220))
                        last_decision = "LOCKED OUT"
                    else:
                        fsm.rollback()
                        pending_candidate = candidate
                        bio_gate.start(cap, candidate.authorization_timeout)
                        add_event(f"BIO REQUIRED: {gesture}", (0, 140, 255))
                        last_decision = "BIO PENDING"

        # ── AUTH_PENDING: auto-trigger biometric ───────────────────────────────
        if fsm.is_state("AUTH_PENDING") and not bio_gate.is_running and pending_candidate is None:
            pending_candidate = TransitionCandidate(
                source_state="AUTH_PENDING",
                gesture="__FACE_AUTH__",
                destination_state="AUTHENTICATED",
                risk_level="MEDIUM",
                biometric_required=True,
                biometric_type="FACE",
                confidence_threshold=BIOMETRIC_CONFIDENCE_THRESHOLD,
                authorization_timeout=AUTHORIZATION_TIMEOUT_SEC,
                action="NONE",
            )
            bio_gate.start(cap, AUTHORIZATION_TIMEOUT_SEC)
            add_event("Face auth started", (0, 140, 255))

        # ── Biometric request overlay ──────────────────────────────────────────
        if bio_gate.is_running:
            draw_biometric_request_overlay(frame, bio_gate.countdown)

        # ── Dashboard ─────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        panel_w = 230
        px = w - panel_w - 4

        draw_state_panel(frame, fsm.current, fsm.previous, fsm.candidate,
                         px, 4, panel_w, 110)
        draw_gesture_panel(frame, gesture or "—", conf, stable_count,
                           STABILITY_FRAMES, px, 118, panel_w, 110)
        draw_auth_panel(frame, auth_mgr.is_authorized(), bio_conf_display,
                        auth_mgr.remaining_seconds(), auth_mgr.is_locked_out(),
                        auth_mgr.lockout_remaining(), auth_mgr.failure_count,
                        px, 232, panel_w, 110)
        draw_adaptive_panel(frame, last_auth_result, escalation.level_name,
                            temporal.binding_score(), authenticity.score(),
                            px, 346, panel_w, 160)
        if last_candidate:
            draw_transition_panel(
                frame, last_candidate.source_state, last_candidate.gesture,
                last_candidate.destination_state, last_candidate.risk_level,
                policy_bio_required(last_candidate), last_decision,
                px, 510, panel_w, 140,
            )
        draw_security_events(frame, security_events, px, 654, panel_w, 180)
        draw_state_graph(frame, fsm.current, all_transitions, px, 838, panel_w, 160)
        draw_fps_and_session(frame, fps, session.session_id, 10, h - 10)

        # ── Top HUD ────────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w - panel_w - 8, 90), (18, 18, 30), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        g_col = COLOURS.get(gesture, (200, 200, 200))
        cv2.putText(frame, f"Gesture: {gesture or '—'}  {conf*100:.1f}%",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, g_col, 2)
        s_col = (0, 200, 80) if fsm.current not in ("LOCKED", "ERROR") else (0, 60, 220)
        mode_tag = "[ADAPTIVE]" if ADAPTIVE_SECURITY else "[CLASSIC]"
        cv2.putText(frame, f"State: {fsm.current}  {mode_tag}",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_col, 2)

        cv2.imshow("Adaptive Biometric-Gated Engine", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            sequence = []
            stable_count = 0
        elif key == ord('l'):
            auth_mgr.revoke()
            token_store.invalidate("manual relock")
            fsm.force("LOCKED")
            add_event("Manual relock", (0, 60, 220))

    # ── Cleanup ────────────────────────────────────────────────────────────────
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    session.close()
    logger.info("Session closed: %s", session.session_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adaptive Biometric-Gated Engine")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=["rf", "mlp", "svm"])
    args, _ = ap.parse_known_args()
    main(args.model)
