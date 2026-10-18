"""
ui/overlays.py
--------------
OpenCV drawing helpers for the security dashboard.
All functions take a frame (numpy array) and return it modified.
No business logic lives here.
"""

import cv2
import numpy as np
import time
import collections
from typing import Optional
from collections import deque

# ── Colour palette (BGR) ───────────────────────────────────────────────────────
C_BG       = (18,  18,  30)
C_PANEL    = (28,  28,  45)
C_BORDER   = (60,  60,  90)
C_WHITE    = (230, 230, 230)
C_DIM      = (110, 110, 140)
C_GREEN    = (0,   200, 80)
C_RED      = (0,   60,  220)
C_YELLOW   = (0,   200, 220)
C_BLUE     = (200, 120, 0)
C_ORANGE   = (0,   140, 255)

RISK_COLOURS = {
    "LOW":      C_GREEN,
    "MEDIUM":   C_YELLOW,
    "HIGH":     C_ORANGE,
    "CRITICAL": C_RED,
}


def _panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
           title: str = "", alpha: float = 0.75) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), C_PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BORDER, 1)
    if title:
        cv2.putText(frame, title, (x1 + 8, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1)


def _text(frame, text: str, x: int, y: int,
          colour=C_WHITE, scale: float = 0.52, thickness: int = 1) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness)


# ── Individual panels ──────────────────────────────────────────────────────────

def draw_state_panel(
    frame: np.ndarray,
    current: str,
    previous: str,
    candidate: Optional[str],
    x: int, y: int, w: int = 220, h: int = 110,
) -> None:
    _panel(frame, x, y, x + w, y + h, "STATE")
    state_col = C_GREEN if current not in ("LOCKED", "ERROR") else C_RED
    _text(frame, current,  x + 8, y + 40, state_col, 0.65, 2)
    _text(frame, f"prev: {previous}", x + 8, y + 62, C_DIM, 0.40)
    if candidate:
        _text(frame, f"next: {candidate}", x + 8, y + 80, C_YELLOW, 0.40)


def draw_gesture_panel(
    frame: np.ndarray,
    gesture: str,
    confidence: float,
    stable_count: int,
    stable_thresh: int,
    x: int, y: int, w: int = 220, h: int = 110,
) -> None:
    _panel(frame, x, y, x + w, y + h, "GESTURE")
    from utils.gesture_config import COLOURS
    col = COLOURS.get(gesture, C_WHITE)
    _text(frame, gesture or "—", x + 8, y + 38, col, 0.60, 2)
    _text(frame, f"conf: {confidence*100:.1f}%", x + 8, y + 58, C_DIM, 0.42)
    _text(frame, f"stab: {stable_count}/{stable_thresh}", x + 8, y + 76, C_DIM, 0.42)
    # stability bar
    bar_w = int((w - 16) * min(stable_count / stable_thresh, 1.0))
    cv2.rectangle(frame, (x + 8, y + 84), (x + w - 8, y + 94), (40, 40, 60), -1)
    cv2.rectangle(frame, (x + 8, y + 84), (x + 8 + bar_w, y + 94), col, -1)


def draw_auth_panel(
    frame: np.ndarray,
    is_authorized: bool,
    bio_confidence: float,
    remaining_sec: float,
    is_locked_out: bool,
    lockout_remaining: float,
    failure_count: int,
    x: int, y: int, w: int = 220, h: int = 130,
) -> None:
    _panel(frame, x, y, x + w, y + h, "BIOMETRIC AUTH")
    if is_locked_out:
        _text(frame, "LOCKED OUT", x + 8, y + 38, C_RED, 0.60, 2)
        _text(frame, f"cooldown: {lockout_remaining:.0f}s", x + 8, y + 58, C_ORANGE, 0.42)
        return
    status_col = C_GREEN if is_authorized else C_RED
    status_txt = "AUTHORIZED" if is_authorized else "NOT AUTH"
    _text(frame, status_txt, x + 8, y + 38, status_col, 0.58, 2)
    _text(frame, f"bio conf: {bio_confidence*100:.1f}%", x + 8, y + 58, C_DIM, 0.42)
    if is_authorized:
        _text(frame, f"expires: {remaining_sec:.1f}s", x + 8, y + 76, C_YELLOW, 0.42)
    _text(frame, f"failures: {failure_count}", x + 8, y + 94, C_DIM, 0.40)


def draw_transition_panel(
    frame: np.ndarray,
    source: str,
    gesture: str,
    destination: str,
    risk: str,
    bio_needed: bool,
    decision: str,
    x: int, y: int, w: int = 220, h: int = 150,
) -> None:
    _panel(frame, x, y, x + w, y + h, "TRANSITION")
    risk_col   = RISK_COLOURS.get(risk, C_WHITE)
    dec_col    = C_GREEN if decision == "ALLOWED" else C_RED
    _text(frame, source,      x + 8, y + 32, C_DIM,   0.42)
    _text(frame, f"  {gesture}", x + 8, y + 50, C_WHITE, 0.48)
    _text(frame, f"  -> {destination}", x + 8, y + 68, C_WHITE, 0.48)
    _text(frame, f"risk: {risk}", x + 8, y + 88, risk_col, 0.42)
    bio_txt = "bio: REQUIRED" if bio_needed else "bio: NOT REQ"
    _text(frame, bio_txt, x + 8, y + 106, C_YELLOW if bio_needed else C_DIM, 0.40)
    _text(frame, decision, x + 8, y + 126, dec_col, 0.58, 2)


def draw_security_events(
    frame: np.ndarray,
    events: deque,
    x: int, y: int, w: int = 220, h: int = 200,
) -> None:
    _panel(frame, x, y, x + w, y + h, "SECURITY EVENTS")
    for i, (ts, msg, col) in enumerate(list(events)[:7]):
        ey = y + 28 + i * 24
        if ey + 16 > y + h:
            break
        _text(frame, ts,  x + 8,      ey, C_DIM,  0.34)
        _text(frame, msg, x + 8, ey + 12, col,    0.38)


def draw_fps_and_session(
    frame: np.ndarray,
    fps: float,
    session_id: str,
    x: int, y: int,
) -> None:
    _text(frame, f"FPS:{fps:.1f}  {session_id}", x, y, C_DIM, 0.38)


def draw_state_graph(
    frame: np.ndarray,
    current_state: str,
    transitions: list[dict],
    x: int, y: int, w: int = 220, h: int = 200,
) -> None:
    """Full state diagram showing all states, highlighting the current one."""
    ALL_STATES = [
        "LOCKED",
        "AUTH_PENDING",
        "AUTHENTICATED",
        "PRESENTATION",
        "SETTINGS",
        "SENSITIVE_MODE",
        "ERROR",
    ]
    PROTECTED = {"LOCKED", "AUTH_PENDING", "SETTINGS", "SENSITIVE_MODE"}

    row_h = 26
    needed_h = 24 + len(ALL_STATES) * row_h + 8
    h = max(h, needed_h)
    _panel(frame, x, y, x + w, y + h, "ALL STATES")

    for i, state in enumerate(ALL_STATES):
        sy = y + 28 + i * row_h
        is_current = state == current_state

        if is_current:
            cv2.rectangle(frame, (x + 4, sy - 14), (x + w - 4, sy + 8), (50, 50, 80), -1)
            col = C_GREEN if state not in ("LOCKED", "ERROR") else C_RED
            prefix = ">> "
            scale, thick = 0.46, 2
        else:
            col = C_ORANGE if state in PROTECTED else C_DIM
            prefix = "   "
            scale, thick = 0.40, 1

        _text(frame, f"{prefix}{state}", x + 8, sy, col, scale, thick)


def draw_biometric_request_overlay(frame: np.ndarray, countdown: float) -> None:
    """Full-frame overlay shown while waiting for face authentication."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cx, cy = w // 2, h // 2
    cv2.putText(frame, "BIOMETRIC VERIFICATION REQUIRED",
                (cx - 260, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_YELLOW, 2)
    cv2.putText(frame, "Look at the camera",
                (cx - 120, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1)
    cv2.putText(frame, f"Timeout: {countdown:.1f}s",
                (cx - 80, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_ORANGE, 1)


def draw_adaptive_panel(
    frame: np.ndarray,
    auth_result,          # AuthorizationResult | None
    security_level: str,
    identity_continuity: float,
    gesture_authenticity: float,
    x: int, y: int, w: int = 220, h: int = 160,
) -> None:
    """Adaptive engine panel — score, decision, risk, continuity, authenticity."""
    _panel(frame, x, y, x + w, y + h, "ADAPTIVE ENGINE")
    if auth_result is None:
        _text(frame, "No decision yet", x + 8, y + 38, C_DIM, 0.42)
    else:
        score = auth_result.score
        score_col = C_GREEN if score >= 0.85 else C_YELLOW if score >= 0.65 else C_RED
        _text(frame, f"Score: {score*100:.1f}%", x + 8, y + 32, score_col, 0.58, 2)
        dec_col = C_GREEN if auth_result.decision == "ALLOW" else C_RED
        _text(frame, auth_result.decision, x + 8, y + 52, dec_col, 0.52, 2)
        _text(frame, f"risk: {auth_result.risk_score:.2f}", x + 8, y + 70, C_DIM, 0.40)
        _text(frame, f"bio thr: {auth_result.biometric_threshold:.2f}", x + 8, y + 86, C_DIM, 0.40)
    _text(frame, f"sec: {security_level}", x + 8, y + 104,
          C_GREEN if security_level == "NORMAL" else C_ORANGE, 0.40)
    _text(frame, f"continuity: {identity_continuity*100:.0f}%", x + 8, y + 120, C_DIM, 0.40)
    _text(frame, f"authenticity: {gesture_authenticity*100:.0f}%", x + 8, y + 136, C_DIM, 0.40)
    # continuity bar
    bar_w = int((w - 16) * identity_continuity)
    cv2.rectangle(frame, (x + 8, y + 144), (x + w - 8, y + 154), (40, 40, 60), -1)
    bar_col = C_GREEN if identity_continuity > 0.7 else C_ORANGE
    cv2.rectangle(frame, (x + 8, y + 144), (x + 8 + bar_w, y + 154), bar_col, -1)
