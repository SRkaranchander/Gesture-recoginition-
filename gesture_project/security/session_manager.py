"""
security/session_manager.py
----------------------------
Generates and manages the runtime session ID.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database import db


class SessionManager:
    def __init__(self) -> None:
        self._session_id: str = self._generate()
        db.create_session(self._session_id)

    @staticmethod
    def _generate() -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"SES-{ts}"

    @property
    def session_id(self) -> str:
        return self._session_id

    def close(self) -> None:
        db.close_session(self._session_id)
