# Experimental Methodology

## Experiment A — Baseline (ADAPTIVE_SECURITY = False)

Run with the original fixed-threshold system:
```bash
python main.py --model rf
# In config/settings.py: ADAPTIVE_SECURITY = False
```

Metrics collected:
- Unauthorized transition rate
- False acceptance rate (FAR)
- False rejection rate (FRR)
- Gesture accuracy
- Authentication latency
- Rollback frequency

## Experiment B — Adaptive System (ADAPTIVE_SECURITY = True)

Run with the adaptive engine:
```bash
python main.py --model rf
# In config/settings.py: ADAPTIVE_SECURITY = True
```

Additional metrics:
- Authorization score distribution
- Step-up authentication frequency
- Security escalation events
- Temporal binding score distribution
- Gesture authenticity score distribution
- Adaptive threshold usage per risk level
- Token validation success/failure rate

## Metric Export

```python
from security.security_feedback import SecurityFeedback
metrics = feedback.export_metrics()
# Returns dict: {"ALLOW": N, "DENY": N, "FAIL_BIO": N, ...}
```

Audit log can be queried directly:
```sql
SELECT authorization_result, COUNT(*) FROM transition_log GROUP BY authorization_result;
SELECT AVG(authorization_score) FROM transition_log WHERE authorization_score IS NOT NULL;
SELECT AVG(biometric_confidence) FROM transition_log WHERE authentication_result='PASS';
```

## Comparison Metrics

| Metric | Baseline | Adaptive |
|--------|----------|---------|
| FAR | Fixed threshold | Risk-adaptive threshold |
| FRR | Fixed threshold | Score-based, context-aware |
| Auth latency | Single check | Score computation + token |
| Unauthorized transitions | Binary gate | Multi-level gate |
| Identity loss response | Timeout-based | Continuous + temporal binding |
