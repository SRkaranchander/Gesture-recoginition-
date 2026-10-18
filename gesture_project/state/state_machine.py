"""
state/state_machine.py
----------------------
Explicit finite-state machine.
States are loaded from the database; no state names are hard-coded here.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StateMachine:
    """
    Holds the current application state and enforces valid transitions.
    The set of valid states is loaded from the DB at startup so new states
    can be added through configuration without touching this file.
    """

    def __init__(self, valid_states: set[str], initial: str = "LOCKED") -> None:
        if initial not in valid_states:
            raise ValueError(f"Initial state '{initial}' not in valid_states")
        self._valid_states: set[str] = valid_states
        self._current: str = initial
        self._previous: str = initial
        self._candidate: Optional[str] = None

    # ── Read-only properties ───────────────────────────────────────────────────

    @property
    def current(self) -> str:
        return self._current

    @property
    def previous(self) -> str:
        return self._previous

    @property
    def candidate(self) -> Optional[str]:
        return self._candidate

    # ── Mutation ───────────────────────────────────────────────────────────────

    def propose(self, destination: str) -> bool:
        """
        Propose a candidate destination state.
        Returns True if the destination is a known state, False otherwise.
        The state is NOT changed here — authorization must happen first.
        """
        if destination not in self._valid_states:
            logger.warning("Unknown destination state: %s", destination)
            return False
        self._candidate = destination
        return True

    def commit(self) -> str:
        """
        Commit the previously proposed candidate state.
        Must only be called after authorization succeeds.
        Returns the new current state.
        """
        if self._candidate is None:
            raise RuntimeError("commit() called with no pending candidate")
        self._previous = self._current
        self._current  = self._candidate
        self._candidate = None
        logger.info("State transition: %s → %s", self._previous, self._current)
        return self._current

    def rollback(self) -> None:
        """Discard the candidate; state remains unchanged."""
        self._candidate = None

    def force(self, state: str) -> None:
        """Force a state change without authorization (used for relock/error)."""
        if state not in self._valid_states:
            logger.error("force() called with unknown state: %s", state)
            return
        self._previous = self._current
        self._current  = state
        self._candidate = None
        logger.info("Forced state: %s → %s", self._previous, self._current)

    def is_state(self, state: str) -> bool:
        return self._current == state
