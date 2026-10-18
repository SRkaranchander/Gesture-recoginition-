"""
security/authorization.py
--------------------------
Manages the temporary authorization window, failed-attempt counter,
and lockout logic.  Does NOT perform biometric verification itself.
"""

import time
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    AUTHORIZATION_TIMEOUT_SEC,
    MAX_AUTH_FAILURES,
    AUTH_LOCKOUT_SEC,
)

logger = logging.getLogger(__name__)


class AuthorizationManager:
    """
    Tracks whether a valid biometric authorization window is currently open,
    handles failed-attempt counting, and enforces lockout.
    """

    def __init__(self) -> None:
        self._auth_time: Optional[float] = None      # when auth succeeded
        self._expiry: Optional[float]    = None      # when it expires
        self._timeout: float = AUTHORIZATION_TIMEOUT_SEC
        self._failures: int  = 0
        self._locked_until: Optional[float] = None
        self._user_id: Optional[int] = None

    # ── Authorization window ───────────────────────────────────────────────────

    def grant(self, user_id: int, timeout: Optional[float] = None) -> None:
        """Open an authorization window for user_id."""
        self._user_id   = user_id
        self._auth_time = time.time()
        self._expiry    = self._auth_time + (timeout or self._timeout)
        self._failures  = 0
        logger.info("Authorization granted to user %s, expires in %.1fs",
                    user_id, timeout or self._timeout)

    def is_authorized(self) -> bool:
        """Return True if a valid, unexpired authorization window is open."""
        if self._expiry is None:
            return False
        if time.time() > self._expiry:
            logger.info("Authorization window expired")
            self._expiry    = None
            self._auth_time = None
            return False
        return True

    def revoke(self) -> None:
        """Immediately close the authorization window."""
        self._expiry    = None
        self._auth_time = None
        self._user_id   = None
        logger.info("Authorization revoked")

    def remaining_seconds(self) -> float:
        """Seconds remaining in the current authorization window (0 if expired)."""
        if self._expiry is None:
            return 0.0
        return max(0.0, self._expiry - time.time())

    @property
    def authorized_user_id(self) -> Optional[int]:
        return self._user_id if self.is_authorized() else None

    # ── Failed-attempt tracking ────────────────────────────────────────────────

    def record_failure(self) -> int:
        """Increment failure counter; trigger lockout if threshold reached."""
        import config.settings as _cfg  # read at call-time so tests can patch
        self._failures += 1
        logger.warning("Auth failure %d/%d", self._failures, _cfg.MAX_AUTH_FAILURES)
        if self._failures >= _cfg.MAX_AUTH_FAILURES:
            self._locked_until = time.time() + _cfg.AUTH_LOCKOUT_SEC
            logger.warning("Lockout triggered for %.0fs", _cfg.AUTH_LOCKOUT_SEC)
        return self._failures

    def reset_failures(self) -> None:
        self._failures = 0

    def is_locked_out(self) -> bool:
        if self._locked_until is None:
            return False
        if time.time() > self._locked_until:
            self._locked_until = None
            self._failures = 0
            return False
        return True

    def lockout_remaining(self) -> float:
        if self._locked_until is None:
            return 0.0
        return max(0.0, self._locked_until - time.time())

    @property
    def failure_count(self) -> int:
        return self._failures
