# Security Model

## Security Invariants

| # | Invariant | Enforcement |
|---|-----------|-------------|
| 1 | Action never executes before authorization | `execute_action()` only called after `fsm.commit()` |
| 2 | State never changes before authorization | `propose()` ≠ `commit()` — commit only after auth |
| 3 | Failed auth leaves state unchanged | `fsm.rollback()` on every denial path |
| 4 | Expired authorization cannot authorize | `auth_mgr.is_authorized()` checks `time.time() > expiry` |
| 5 | Identity loss invalidates protected auth | `token_store.invalidate()` + `auth_mgr.revoke()` on loss |
| 6 | Token cannot be reused for different transition | Token validates source, gesture, destination, user |
| 7 | Repeated failures force LOCKED | `auth_mgr.record_failure()` → lockout → `fsm.force("LOCKED")` |
| 8 | Every attempt is auditable | `audit.log()` called on every path including denials |
| 9 | Unauthenticated context cannot execute protected action | Biometric gate blocks commit until auth succeeds |
| 10 | Escalation cannot bypass state machine | Escalation only affects score/thresholds, not FSM logic |

## Threat Model

| Threat | Mitigation |
|--------|-----------|
| Gesture replay | Gesture authenticity scorer (velocity, duration, trajectory) |
| Identity substitution | Temporal binding — face must match during gesture window |
| Brute-force biometric | Failure counter + lockout + security escalation |
| Stale authorization | Adaptive timeout per risk level; CRITICAL = 0s (always re-auth) |
| Unauthorized user at camera | Continuous authentication + auto-relock on identity loss |
| Token reuse | Single-use token consumed on first valid use |

## Privacy

- Raw webcam frames are processed transiently and never stored.
- Raw face images are not stored; only the averaged LBP embedding is persisted.
- Audit logs record authentication outcomes, not biometric data.
- Biometric embeddings are stored as JSON float arrays in local SQLite only.
