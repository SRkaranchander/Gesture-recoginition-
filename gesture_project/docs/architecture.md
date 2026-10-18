# System Architecture

## Layer Overview

```
CAMERA
  ├── GESTURE PERCEPTION (MediaPipe → 78-D features → ML classifier → stability filter)
  └── BIOMETRIC PERCEPTION (Haar cascade → LBP embedding → cosine similarity)
          │
     CONTEXT FUSION (AuthorizationContext dataclass)
          │
     TRANSITION PROPOSER (find_transition → TransitionCandidate)
          │
     RISK ENGINE (risk_scores mapping → numerical risk_score)
          │
     ADAPTIVE AUTHORIZATION (compute_score → decide → AuthorizationResult)
          │
    ┌─────┼──────┐
  ALLOW  STEP-UP  DENY/LOCK
    │       │
  TOKEN   BIO GATE (background thread)
  ISSUE      │
    │    TOKEN ISSUE
    └────┬───┘
    TOKEN VALIDATE
          │
     COMMIT / ROLLBACK
          │
     ACTION EXECUTOR
          │
     AUDIT LOGGER → SQLite
          │
     DASHBOARD (OpenCV overlays)
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `utils/gesture_config.py` | Feature extraction, gesture labels, drawing helpers |
| `state/state_machine.py` | FSM — propose/commit/rollback, never changes state before auth |
| `state/transition_manager.py` | DB lookup → TransitionCandidate |
| `security/policy_engine.py` | Risk level → biometric required (original path) |
| `security/adaptive_authorization.py` | Scoring engine, multi-level decision |
| `security/temporal_binding.py` | Identity continuity tracking during gesture window |
| `security/authorization_token.py` | Transition-bound single-use tokens |
| `security/security_escalation.py` | Dynamic security level (0–4) |
| `security/gesture_authenticity.py` | Heuristic temporal gesture analysis |
| `security/security_feedback.py` | Outcome → escalation feedback loop |
| `security/authorization.py` | Auth window, failure counter, lockout |
| `biometric/face_auth.py` | Haar + LBP enrollment and authentication |
| `audit_log/audit_logger.py` | Structured audit log to SQLite |
| `database/db.py` | Schema, migration, all SQL |
| `ui/overlays.py` | OpenCV dashboard panels |
| `main.py` | Frame loop, wires all modules, ADAPTIVE_SECURITY flag |
