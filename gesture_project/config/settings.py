"""
config/settings.py
------------------
All tunable constants for the Biometric-Gated State Transition Engine.
No security values are hard-coded in any other module.
"""

# ── Gesture recognition ────────────────────────────────────────────────────────
GESTURE_CONFIDENCE_THRESHOLD: float = 0.80   # minimum confidence to accept a gesture
STABILITY_FRAMES: int = 8                     # frames a gesture must hold before firing
GESTURE_COOLDOWN_SEC: float = 1.5            # seconds before same gesture can fire again

# ── Biometric ──────────────────────────────────────────────────────────────────
BIOMETRIC_CONFIDENCE_THRESHOLD: float = 0.55  # face-match cosine similarity threshold
ENROLLMENT_SAMPLES: int = 10                  # face frames captured during enrollment
CONTINUOUS_AUTH: bool = True                  # re-verify identity every N seconds
CONTINUOUS_AUTH_INTERVAL_SEC: float = 3.0    # how often to re-check identity
CONTINUOUS_AUTH_LOSS_TIMEOUT_SEC: float = 5.0 # seconds of identity loss before relock

# ── Authorization window ───────────────────────────────────────────────────────
AUTHORIZATION_TIMEOUT_SEC: float = 15.0      # seconds a biometric auth stays valid
MAX_AUTH_FAILURES: int = 3                    # failed attempts before lockout
AUTH_LOCKOUT_SEC: float = 30.0               # lockout duration after max failures

# ── Risk policy — which risk levels require biometric ─────────────────────────
# Values: "NEVER" | "ALWAYS" | "CONFIGURABLE"
RISK_POLICY: dict = {
    "LOW":      "NEVER",
    "MEDIUM":   "CONFIGURABLE",   # follows biometric_required flag in transition row
    "HIGH":     "ALWAYS",
    "CRITICAL": "ALWAYS",
}

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH: str = "data/biometric_gate.db"

# ── Model ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL: str = "rf"   # rf | mlp | svm

# ── UI ────────────────────────────────────────────────────────────────────────
DASHBOARD_WIDTH: int  = 1400
DASHBOARD_HEIGHT: int = 780
SECURITY_EVENT_HISTORY: int = 8   # events shown in the security panel

# ── Adaptive Security Engine ───────────────────────────────────────────────────
# Set ADAPTIVE_SECURITY = False to fall back to the original fixed-threshold system.
ADAPTIVE_SECURITY: bool = False

ADAPTIVE_SECURITY_CONFIG: dict = {
    # Multi-level authorization thresholds (score in [0,1])
    "authorization": {
        "allow_threshold":  0.85,   # score >= this → ALLOW
        "reauth_threshold": 0.65,   # score >= this → REAUTHENTICATE
        "stepup_threshold": 0.45,   # score >= this → STEP_UP
        "deny_threshold":   0.25,   # score >= this → DENY  (below → LOCK)
    },
    # Scoring weights (must sum to ≤ 1.0 for positive terms)
    "weights": {
        "gesture":            0.20,
        "biometric":          0.30,
        "stability":          0.10,
        "identity_continuity":0.20,
        "risk":               0.10,   # subtracted
        "failure_history":    0.05,   # subtracted
        "auth_age":           0.05,   # subtracted
    },
    # Risk → numerical score mapping
    "risk_scores": {
        "LOW":      0.10,
        "MEDIUM":   0.40,
        "HIGH":     0.70,
        "CRITICAL": 0.90,
    },
    # Adaptive biometric thresholds per risk level
    "biometric": {
        "low_threshold":      0.55,
        "medium_threshold":   0.65,
        "high_threshold":     0.75,
        "critical_threshold": 0.85,
    },
    # Adaptive authorization window durations (seconds) per risk level
    "authorization_window": {
        "LOW":      30.0,
        "MEDIUM":   15.0,
        "HIGH":      8.0,
        "CRITICAL":  0.0,   # 0 = always re-authenticate
    },
    # Security escalation
    "security_escalation": {
        "enabled":      True,
        "max_failures": 3,
        "deescalate_after_sec": 120.0,  # seconds of clean activity before de-escalation
    },
    # Temporal binding
    "temporal_binding": {
        "enabled":          True,
        "max_identity_gap": 1.0,   # seconds face may be absent during gesture
    },
    # Gesture authenticity
    "gesture_authenticity": {
        "enabled":             True,
        "min_duration_sec":    0.20,   # gesture must last at least this long
        "max_velocity_change": 3.0,    # max normalised velocity change between frames
    },
}
