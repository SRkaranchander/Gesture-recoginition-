# Technical Contribution Feature Mapping

This document maps each proposed technical contribution to its implementation.
No patentability claims are made. These are descriptions of the implemented prototype.

## Existing Components (Prior Art)

| Component | Implementation |
|-----------|---------------|
| Gesture recognition | MediaPipe + RF/SVM/MLP classifiers |
| Face recognition | OpenCV Haar cascade + LBP histogram |
| State machine | `state/state_machine.py` |
| SQLite audit log | `database/db.py` |

## Proposed Technical Contributions

### 1. Adaptive Transition Authorization Score
**File:** `security/adaptive_authorization.py`
A configurable weighted scoring function combining gesture confidence,
biometric confidence, identity continuity, gesture authenticity, risk score,
failure history, and authentication age into a normalized [0,1] score.

### 2. Multi-Level Authorization Decision
**File:** `security/adaptive_authorization.py`
Maps the authorization score to five decision levels (ALLOW, REAUTHENTICATE,
STEP_UP, DENY, LOCK) with configurable thresholds.

### 3. Transition-Specific Authorization Token
**File:** `security/authorization_token.py`
A short-lived, single-use token bound to a specific
(user, source_state, gesture, destination_state) tuple. Cannot be reused
for a different transition.

### 4. Temporal Biometric-Gesture Binding
**File:** `security/temporal_binding.py`
Tracks whether the authenticated identity was continuously present during
the gesture execution window. Computes a TemporalBindingScore that feeds
into the authorization score.

### 5. Dynamic Risk Assessment
**File:** `security/adaptive_authorization.py` + `config/settings.py`
Numerical risk scores per risk level, used in the authorization formula
rather than binary biometric-required flags.

### 6. Adaptive Biometric Threshold
**File:** `security/adaptive_authorization.py`
Biometric confidence threshold varies by risk level (LOW=0.55 to CRITICAL=0.85)
rather than using a single fixed threshold.

### 7. Adaptive Authorization Lifetime
**File:** `security/adaptive_authorization.py` + `security/authorization.py`
Authorization window duration varies by risk level (LOW=30s to CRITICAL=0s).

### 8. Security Escalation State Machine
**File:** `security/security_escalation.py`
Dynamic security level (0–4) that rises on suspicious events and falls
after configurable clean-activity periods.

### 9. Gesture Authenticity Analysis
**File:** `security/gesture_authenticity.py`
Heuristic temporal analysis of gesture duration, velocity variance, and
joint-angle trajectory consistency to detect anomalous gesture patterns.

### 10. Security Feedback Loop
**File:** `security/security_feedback.py`
Outcome events (allow, deny, failed bio, identity loss) feed back into
the security escalation engine, affecting future authorization decisions.
