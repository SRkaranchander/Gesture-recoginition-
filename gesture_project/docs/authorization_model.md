# Authorization Model

## Adaptive Authorization Score

```
score = w_gesture  * gesture_confidence
      + w_biometric * biometric_confidence
      + w_stability * stability_ratio
      + w_identity  * identity_continuity
      + w_gesture   * gesture_authenticity * 0.5
      - w_risk      * risk_score
      - w_failure   * failure_penalty
      - w_age       * age_penalty
      - escalation_level * 0.10
```

All weights are in `ADAPTIVE_SECURITY_CONFIG["weights"]`. Score is clipped to [0, 1].

## Decision Thresholds

| Score Range | Decision | Meaning |
|-------------|----------|---------|
| >= 0.85 | ALLOW | Immediate authorization |
| 0.65 – 0.85 | REAUTHENTICATE | Re-verify biometric |
| 0.45 – 0.65 | STEP_UP | Enhanced verification required |
| 0.25 – 0.45 | DENY | Transition rejected |
| < 0.25 | LOCK | Force system lock |

All thresholds are configurable in `ADAPTIVE_SECURITY_CONFIG["authorization"]`.

## Adaptive Biometric Thresholds

| Risk Level | Default Threshold |
|------------|------------------|
| LOW | 0.55 |
| MEDIUM | 0.65 |
| HIGH | 0.75 |
| CRITICAL | 0.85 |

## Adaptive Authorization Windows

| Risk Level | Default Window |
|------------|---------------|
| LOW | 30 seconds |
| MEDIUM | 15 seconds |
| HIGH | 8 seconds |
| CRITICAL | 0 seconds (always re-authenticate) |

## Authorization Token

A token is issued after successful authentication and is bound to exactly one
`(user_id, source_state, gesture, destination_state)` tuple. It is single-use
and expires after the adaptive authorization window. Token validation is the
final gate before `fsm.commit()` is called.

## Security Escalation Levels

| Level | Name | Effect |
|-------|------|--------|
| 0 | NORMAL | Standard thresholds |
| 1 | SUSPICIOUS | Score penalty -0.10 |
| 2 | ELEVATED | Score penalty -0.20 |
| 3 | RESTRICTED | Score penalty -0.30 |
| 4 | LOCKED | System locked |

Escalation rises on: failed auth, identity loss, repeated denials, gesture anomaly.
De-escalation occurs after `deescalate_after_sec` of clean activity.
