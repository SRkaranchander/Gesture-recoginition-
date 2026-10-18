# Biometric-Gated State Transition Engine for Gesture-Controlled Interfaces

## 1. Existing System

The base system is a real-time **Dynamic Hand Gesture Recognition** pipeline for
Human-Robot Interaction and Gesture-Based Presentation Control.

| Component | Detail |
|-----------|--------|
| Input | Webcam (OpenCV) |
| Hand tracking | Google MediaPipe (21 landmarks) |
| Feature vector | 78-D (63 normalised landmarks + 15 joint angles) |
| Temporal window | 30 frames |
| Classifiers | Random Forest · SVM · MLP |
| Stability filter | 8-frame confirmation |
| System actions | PyAutoGUI |

**10 recognised gestures:** Start, Stop, Turn Left, Turn Right, Upward, Downward,
Turn Up, Turn Down, Grab, Release.

---

## 2. New Invention — Biometric-Gated State Transition Engine

The proposed system extends the gesture pipeline with a security and
authorization layer. A gesture is no longer directly mapped to an action.
Instead, it **proposes a state transition** that must pass through a
risk-aware policy engine and, for protected operations, a biometric gate.

> **Central principle:** Gesture ≠ Authorization.
> The gesture proposes. The biometric authorizes. The state machine decides.

---

## 3. System Architecture

```
CAMERA
  ↓ MediaPipe hand detection
LANDMARK EXTRACTION (21 landmarks)
  ↓ full_feature_vector()
78-D FEATURE VECTOR
  ↓ 30-frame buffer
GESTURE CLASSIFIER (RF / MLP / SVM)
  ↓ predict_proba()
GESTURE CONFIDENCE
  ↓ 8-frame stability filter
GESTURE EVENT
  ↓ find_transition()
TRANSITION LOOKUP (SQLite)
  ↓ policy_engine.biometric_required()
RISK / POLICY EVALUATION
  ↓
BIOMETRIC GATE ──────────────────────────────────────────────┐
  ↓ face_auth.authenticate()                                  │
AUTHORIZATION DECISION                                        │
  ↓ (ALLOWED)          ↓ (DENIED)                            │
STATE TRANSITION    ROLLBACK + AUDIT LOG ←───────────────────┘
  ↓
ACTION EXECUTOR (system_actions.execute())
  ↓
AUDIT LOG (SQLite transition_log)
  ↓
DASHBOARD UPDATE
```

---

## 4. State Machine

States are stored in the `states` table and loaded at runtime.
New states can be added through the database without modifying code.

| State | Protected | Description |
|-------|-----------|-------------|
| LOCKED | ✓ | System locked |
| AUTH_PENDING | ✓ | Biometric auth in progress |
| AUTHENTICATED | — | User verified |
| PRESENTATION | — | Presentation control |
| SETTINGS | ✓ | Settings — requires auth |
| SENSITIVE_MODE | ✓ | Every action gated |
| ERROR | — | Error state |

---

## 5. Biometric Gate

Face authentication uses OpenCV's built-in Haar cascade detector and an
LBP histogram embedding. No external `face_recognition` / `dlib` dependency
is required.

**Interface (`biometric/face_auth.py`):**
- `enroll_user(cap, username)` — capture samples, store embedding
- `verify_frame(frame)` → `(user_id, confidence)`
- `authenticate(cap, timeout)` → `(user_id, confidence)`
- `is_face_present(frame)` → `bool`

---

## 6. Transition Policy

The full transition table lives in the `transitions` SQLite table.
Each row defines:

```json
{
  "source_state": "PRESENTATION",
  "gesture": "Stop",
  "destination_state": "LOCKED",
  "risk_level": "HIGH",
  "biometric_required": true,
  "biometric_type": "FACE",
  "confidence_threshold": 0.90,
  "timeout": 15.0,
  "action": "STOP_PRESENTATION"
}
```

---

## 7. Risk Levels

| Level | Examples | Biometric policy |
|-------|----------|-----------------|
| LOW | Next/Prev Slide, Volume | Never required |
| MEDIUM | Start Presentation, Settings nav | Follows transition flag |
| HIGH | Stop Presentation, Open Settings | Always required |
| CRITICAL | Sensitive operations | Always required |

Policy is configured in `config/settings.py → RISK_POLICY`.

---

## 8. Authentication Mechanism

- **One-time:** authenticate once, then re-authenticate for each protected action.
- **Temporary window:** after successful auth, a configurable window
  (`AUTHORIZATION_TIMEOUT_SEC`) allows subsequent protected transitions
  without re-authenticating.
- **Continuous:** if `CONTINUOUS_AUTH=True`, identity is re-verified every
  `CONTINUOUS_AUTH_INTERVAL_SEC` seconds. Identity loss triggers automatic relock.

---

## 9. Database

SQLite file: `data/biometric_gate.db`

Tables: `users`, `biometric_profiles`, `states`, `transitions`,
`sessions`, `transition_log`, `system_config`.

---

## 10. Security Model

- Raw webcam frames are processed transiently and never stored.
- Raw face images are not stored; only the averaged LBP embedding is persisted.
- Biometric embeddings are stored as JSON float arrays in the local SQLite DB.
- Audit logs record authentication outcomes, not biometric data.
- Failed authentication triggers a configurable lockout (`MAX_AUTH_FAILURES`,
  `AUTH_LOCKOUT_SEC`).
- Authorization windows expire automatically (`AUTHORIZATION_TIMEOUT_SEC`).

---

## 11. Installation

```bash
pip install -r requirements.txt
```

No additional dependencies beyond the existing project requirements.
`pytest` is added for testing only.

---

## 12. Running the System

### Step 1 — Enroll your face (once)

```bash
cd gesture_project
python enroll.py --username alice
```

### Step 2 — Run the engine

```bash
cd /d "d:\TOC project\gesture_project"

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

python enroll.py --username alice

python main.py --model rf
```

### Original Demo (unchanged)

```bash
python Demo.py --model mlp
```

### Keyboard shortcuts (main.py)

| Key | Action |
|-----|--------|
| Q | Quit |
| R | Reset gesture buffer |
| L | Manual relock |

---

## 13. Configuration

All security values are in `config/settings.py`:

```python
GESTURE_CONFIDENCE_THRESHOLD  = 0.80
BIOMETRIC_CONFIDENCE_THRESHOLD = 0.55
STABILITY_FRAMES               = 8
AUTHORIZATION_TIMEOUT_SEC      = 15.0
MAX_AUTH_FAILURES              = 3
AUTH_LOCKOUT_SEC               = 30.0
CONTINUOUS_AUTH                = True
CONTINUOUS_AUTH_INTERVAL_SEC   = 3.0
```

---

## 14. Testing

```bash
python -m pytest tests/test_engine.py -v
```

Test categories:
- State machine (valid/invalid transitions, rollback, force)
- Transition manager (lookup, confidence threshold, risk level)
- Policy engine (risk → biometric requirement)
- Authorization manager (window, expiry, lockout, failure counter)
- Security integration (unauthorized gesture, full authorized flow,
  expired window, audit log)

---

## 15. Demonstration Scenario

| Step | Action | State |
|------|--------|-------|
| 1 | System starts | LOCKED |
| 2 | User performs **Start** | AUTH_PENDING |
| 3 | Face verified | AUTHENTICATED |
| 4 | User performs **Start** again | PRESENTATION |
| 5 | **Turn Right** → Next Slide (LOW risk, no bio) | PRESENTATION |
| 6 | **Stop** → biometric required | PRESENTATION (pending) |
| 7 | Correct user authenticates | LOCKED |
| 8 | Unauthorized person attempts **Stop** → bio fails | PRESENTATION (unchanged) |

Audit log records `UNAUTHORIZED TRANSITION DENIED` for step 8.

---

## 16. Limitations

- LBP histogram embedding is a lightweight prototype; it is less robust than
  deep-learning face embeddings (e.g., FaceNet, ArcFace) under varying lighting.
- The biometric authentication runs in a background thread sharing the same
  `VideoCapture` object; on some platforms this may cause frame contention.
- The implemented prototype has not undergone formal security evaluation.

---

## 17. Future Work

- Replace LBP embedding with a deep face embedding (FaceNet / InsightFace).
- Add liveness detection to prevent photo-based spoofing.
- Support multi-user enrollment and per-user transition policies.
- Add a web-based administration dashboard for managing transitions and users.
- Evaluate FAR / FRR under controlled conditions.
- Extend to multi-modal biometrics (voice + face).
