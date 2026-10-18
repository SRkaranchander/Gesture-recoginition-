"""
utils/gesture_config.py
-----------------------
Central config: gesture labels, landmark indices, colours, and
helper functions used across all modules.
"""

import numpy as np

# ── 10 Gestures (mirrors the paper's HRI gesture set) ─────────────────────────
GESTURE_LABELS = {
    0: "Start",        # Open palm facing camera
    1: "Stop",         # Closed fist
    2: "Turn Left",    # Index pointing left
    3: "Turn Right",   # Index pointing right
    4: "Upward",       # Index pointing up
    5: "Downward",     # Index pointing down
    6: "Turn Up",      # Thumb up
    7: "Turn Down",    # Thumb down
    8: "Grab",         # Fingers curled inward (claw)
    9: "Release",      # Fingers spread wide open
}

GESTURE_ACTIONS = {
    "Start":       "▶  PLAY / START",
    "Stop":        "⏹  STOP / PAUSE",
    "Turn Left":   "⏮  PREVIOUS SLIDE",
    "Turn Right":  "⏭  NEXT SLIDE",
    "Upward":      "🔊  VOLUME UP",
    "Downward":    "🔉  VOLUME DOWN",
    "Turn Up":     "👍  LIKE / CONFIRM",
    "Turn Down":   "👎  DISLIKE / CANCEL",
    "Grab":        "✊  GRAB / SELECT",
    "Release":     "🖐  RELEASE / DESELECT",
}

# ── Colours (BGR for OpenCV) ───────────────────────────────────────────────────
COLOURS = {
    "Start":       (0,   200,  50),
    "Stop":        (0,   0,   220),
    "Turn Left":   (220, 120,  0),
    "Turn Right":  (0,  120,  220),
    "Upward":      (0,  220,  220),
    "Downward":    (220, 0,   220),
    "Turn Up":     (0,  180,  0),
    "Turn Down":   (0,  0,   180),
    "Grab":        (180, 80,   0),
    "Release":     (80,  180,  0),
    "default":     (255, 255, 255),
}

# MediaPipe 21 landmarks per hand
NUM_LANDMARKS = 21
# Each landmark has x, y, z → 63 raw features per frame
RAW_FEATURE_DIM = NUM_LANDMARKS * 3

# Sequence length (frames) for LSTM input
SEQUENCE_LENGTH = 30


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_landmarks(hand_landmarks) -> np.ndarray:
    """
    Extract normalised (x, y, z) for all 21 landmarks.
    Returns shape (63,)
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalise_landmarks(raw: np.ndarray) -> np.ndarray:
    """
    Translate so wrist (landmark 0) is origin, then scale by
    the distance from wrist to middle-finger MCP (landmark 9)
    so the features are scale-invariant.
    Returns shape (63,)
    """
    pts = raw.reshape(NUM_LANDMARKS, 3)
    # Wrist = landmark 0
    wrist = pts[0].copy()
    pts -= wrist
    # Scale by wrist-to-middle-MCP distance
    ref_dist = np.linalg.norm(pts[9])
    if ref_dist > 1e-6:
        pts /= ref_dist
    return pts.flatten()


def landmarks_to_angles(raw: np.ndarray) -> np.ndarray:
    """
    Compute finger-bend angles (dot-product between consecutive
    bone vectors) for all 4 joints of each of the 5 fingers.
    Returns shape (20,) — a compact pose descriptor.
    """
    pts = raw.reshape(NUM_LANDMARKS, 3)

    # Finger landmark indices: [MCP, PIP, DIP, TIP]
    fingers = [
        [1, 2, 3, 4],    # Thumb
        [5, 6, 7, 8],    # Index
        [9, 10, 11, 12], # Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20],# Pinky
    ]

    angles = []
    for finger in fingers:
        for j in range(len(finger) - 2):
            a = pts[finger[j]]
            b = pts[finger[j + 1]]
            c = pts[finger[j + 2]]
            ba = a - b
            bc = c - b
            cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angles.append(np.clip(cos_a, -1, 1))
    return np.array(angles, dtype=np.float32)  # shape (15,) actually

    # Pad / trim to 20
    arr = np.array(angles, dtype=np.float32)
    return np.pad(arr, (0, max(0, 20 - len(arr))))[:20]


def full_feature_vector(hand_landmarks) -> np.ndarray:
    """
    Combines normalised landmarks (63) + angles (~15) → ~78-d vector.
    Used as per-frame feature for the classifier.
    """
    raw  = extract_landmarks(hand_landmarks)
    norm = normalise_landmarks(raw)
    angs = landmarks_to_angles(raw)
    return np.concatenate([norm, angs])


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_info_box(frame, gesture: str, confidence: float, action: str):
    """Draw a styled overlay with gesture name, confidence and action."""
    import cv2
    h, w = frame.shape[:2]
    colour = COLOURS.get(gesture, COLOURS["default"])

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Gesture name
    cv2.putText(frame, f"Gesture: {gesture}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    # Confidence bar
    bar_w = int((w - 200) * confidence)
    cv2.rectangle(frame, (15, 50), (w - 185, 65), (60, 60, 60), -1)
    cv2.rectangle(frame, (15, 50), (15 + bar_w, 65), colour, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (w - 175, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    # Action
    cv2.putText(frame, action,
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    return frame


def draw_landmark_skeleton(frame, hand_landmarks, mp_hands, mp_drawing,
                            gesture: str = "default"):
    """Draw coloured skeleton on the frame."""
    import cv2
    colour = COLOURS.get(gesture, COLOURS["default"])
    style  = mp_drawing.DrawingSpec(color=colour, thickness=2, circle_radius=3)
    conn   = mp_drawing.DrawingSpec(color=colour, thickness=2)
    mp_drawing.draw_landmarks(
        frame, hand_landmarks,
        mp_hands.HAND_CONNECTIONS, style, conn)
    return frame
