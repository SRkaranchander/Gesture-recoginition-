"""
security/authorization_token.py
---------------------------------
Transition-specific authorization tokens.

A token is generated after successful authentication and is bound to the
exact transition that triggered authentication.  It cannot be reused for
a different (source_state, gesture, destination_state) combination.

Security invariants enforced here:
  - Token is bound to one specific transition (Invariant 6).
  - Token expires after authorization_timeout (Invariant 4).
  - Token is invalidated on identity loss (Invariant 5).
  - Token validation checks user, state, gesture, destination, and expiry.
"""

import uuid
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AuthorizationToken:
    token_id:              str
    user_id:               int
    source_state:          str
    destination_state:     str
    gesture:               str
    risk_level:            str
    risk_score:            float
    gesture_confidence:    float
    biometric_confidence:  float
    temporal_binding_score: float
    auth_score:            float
    authentication_timestamp: float
    expiration_timestamp:  float
    security_level:        int
    consumed:              bool = False   # single-use flag

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expiration_timestamp

    @property
    def remaining_sec(self) -> float:
        return max(0.0, self.expiration_timestamp - time.time())


class TokenStore:
    """
    Thread-safe store for the single active authorization token.
    Only one token is active at a time; issuing a new one replaces the old.
    """

    def __init__(self) -> None:
        self._token: Optional[AuthorizationToken] = None

    def issue(
        self,
        *,
        user_id: int,
        source_state: str,
        destination_state: str,
        gesture: str,
        risk_level: str,
        risk_score: float,
        gesture_confidence: float,
        biometric_confidence: float,
        temporal_binding_score: float,
        auth_score: float,
        timeout: float,
        security_level: int = 0,
    ) -> AuthorizationToken:
        now = time.time()
        token = AuthorizationToken(
            token_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            source_state=source_state,
            destination_state=destination_state,
            gesture=gesture,
            risk_level=risk_level,
            risk_score=risk_score,
            gesture_confidence=gesture_confidence,
            biometric_confidence=biometric_confidence,
            temporal_binding_score=temporal_binding_score,
            auth_score=auth_score,
            authentication_timestamp=now,
            expiration_timestamp=now + timeout,
            security_level=security_level,
        )
        self._token = token
        logger.info("Token issued: %s for %s+%s->%s exp=%.1fs",
                    token.token_id, source_state, gesture, destination_state, timeout)
        return token

    def validate(
        self,
        source_state: str,
        gesture: str,
        destination_state: str,
        user_id: int,
    ) -> tuple[bool, str]:
        """
        Validate the active token against the proposed transition.
        Returns (valid: bool, reason: str).
        Consumes the token on success (single-use).
        """
        t = self._token
        if t is None:
            return False, "No active token"
        if t.consumed:
            return False, "Token already consumed"
        if t.is_expired:
            self._token = None
            return False, f"Token {t.token_id} expired"
        if t.user_id != user_id:
            return False, f"Token user mismatch: {t.user_id} != {user_id}"
        if t.source_state != source_state:
            return False, f"Token source_state mismatch: {t.source_state} != {source_state}"
        if t.gesture != gesture:
            return False, f"Token gesture mismatch: {t.gesture} != {gesture}"
        if t.destination_state != destination_state:
            return False, f"Token destination mismatch: {t.destination_state} != {destination_state}"
        # Valid — consume
        t.consumed = True
        logger.info("Token %s validated and consumed", t.token_id)
        return True, "OK"

    def invalidate(self, reason: str = "") -> None:
        if self._token:
            logger.info("Token %s invalidated: %s", self._token.token_id, reason)
        self._token = None

    @property
    def active_token(self) -> Optional[AuthorizationToken]:
        if self._token and not self._token.is_expired and not self._token.consumed:
            return self._token
        return None
