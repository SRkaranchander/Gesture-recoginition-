"""
actions/system_actions.py
--------------------------
Executes system-level actions ONLY after authorization has been granted.
Wraps the existing pyautogui mappings from Demo.py.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    _PYAUTOGUI = True
except ImportError:
    _PYAUTOGUI = False
    logger.warning("pyautogui not available — actions will be logged only")

# Maps action name (from transitions table) → callable
_ACTION_MAP: dict = {}


def _build_map() -> None:
    if not _PYAUTOGUI:
        return
    global _ACTION_MAP
    _ACTION_MAP = {
        "PREV_SLIDE":        lambda: pyautogui.press("left"),
        "NEXT_SLIDE":        lambda: pyautogui.press("right"),
        "VOLUME_UP":         lambda: pyautogui.hotkey("volumeup"),
        "VOLUME_DOWN":       lambda: pyautogui.hotkey("volumedown"),
        "CONFIRM":           lambda: pyautogui.press("up"),
        "CANCEL":            lambda: pyautogui.press("down"),
        "START_PRESENTATION":lambda: pyautogui.press("space"),
        "STOP_PRESENTATION": lambda: pyautogui.press("space"),
        "OPEN_SETTINGS":     lambda: None,   # application-level
        "CLOSE_SETTINGS":    lambda: None,
        "SETTINGS_NEXT":     lambda: pyautogui.press("right"),
        "SETTINGS_PREV":     lambda: pyautogui.press("left"),
        "SETTINGS_CONFIRM":  lambda: pyautogui.press("enter"),
        "ENTER_SENSITIVE":   lambda: None,
        "EXIT_SENSITIVE":    lambda: None,
        "LEAVE_SENSITIVE":   lambda: None,
        "SENSITIVE_NEXT":    lambda: pyautogui.press("right"),
        "SENSITIVE_PREV":    lambda: pyautogui.press("left"),
        "REQUEST_AUTH":      lambda: None,   # handled by biometric module
        "NONE":              lambda: None,
    }


_build_map()


def execute(action: str) -> bool:
    """
    Execute the named action.
    Returns True if the action was dispatched, False if unknown.
    NEVER called before authorization.
    """
    fn = _ACTION_MAP.get(action)
    if fn is None:
        logger.warning("Unknown action: %s", action)
        return False
    try:
        fn()
        logger.info("Action executed: %s", action)
        return True
    except Exception as exc:
        logger.error("Action '%s' failed: %s", action, exc)
        return False
