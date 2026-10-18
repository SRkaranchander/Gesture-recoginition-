#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import pickle, os, sys, time, argparse, collections

if not hasattr(mp, "solutions"):
    sys.exit(
        "\n[ERROR] Unsupported MediaPipe version in this Python environment.\n"
        "Please run this project using its dedicated virtual environment:\n"
        "  ..\\.venv\\Scripts\\python.exe Demo.py\n"
        "or activate the virtual environment first:\n"
        "  ..\\.venv\\Scripts\\Activate.ps1\n"
    )

sys.path.insert(0, os.getcwd())
from utils.gesture_config import (
    GESTURE_LABELS, SEQUENCE_LENGTH,
    full_feature_vector,
    draw_info_box, draw_landmark_skeleton, COLOURS,
    GESTURE_ACTIONS, RAW_FEATURE_DIM
)

# Try importing pyautogui for actual system control
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

MODEL_DIR  = "models"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.65,
)


def load_model(model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    if not os.path.exists(path):
        print(f"[ERROR] Model not found: {path}")
        print("  → Run train_model.py first.")
        sys.exit(1)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"✓ Loaded model: {path}")
    return bundle["model"], bundle["scaler"], bundle["feat"]


def predict(model, scaler, feat_type, sequence: list):

    if len(sequence) < SEQUENCE_LENGTH:
        return None, 0.0

    arr = np.array(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)

    if feat_type == "full":
        X = arr.flatten().reshape(1, -1)
    else:
        X = np.concatenate([arr.mean(axis=0), arr.std(axis=0)]).reshape(1, -1)

    X_scaled = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        gid   = int(np.argmax(proba))
        conf  = float(proba[gid])
    else:
        gid  = int(model.predict(X_scaled)[0])
        conf = 1.0

    return GESTURE_LABELS.get(gid, "Unknown"), conf


def trigger_action(gesture: str):
    """Optionally trigger real system actions via pyautogui."""
    if not PYAUTOGUI_AVAILABLE: 
        return
    actions = {
        "Start":      lambda: pyautogui.press("space"),
        "Stop":       lambda: pyautogui.press("space"),
        "Turn Left":  lambda: pyautogui.press("left"),
        "Turn Right": lambda: pyautogui.press("right"),
        "Upward":     lambda: pyautogui.hotkey("volumeup"),
        "Downward":   lambda: pyautogui.hotkey("volumedown"),
        "Turn Up":    lambda: pyautogui.press("up"),
        "Turn Down":  lambda: pyautogui.press("down"),
    }
    if gesture in actions:
        actions[gesture]()


def draw_gesture_history(frame, history: collections.deque):
    """Draw a log of the last 6 gestures on the right side."""
    h, w = frame.shape[:2]
    panel_x = w - 220

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 5, 110), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "HISTORY", (panel_x, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 180), 1)

    for i, (ts, name) in enumerate(reversed(list(history))):
        y  = 162 + i * 38
        col = COLOURS.get(name, (200, 200, 200))
        alpha = max(0.3, 1.0 - i * 0.15)
        col = tuple(int(c * alpha) for c in col)
        cv2.putText(frame, f"• {name}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)
        cv2.putText(frame, ts, (panel_x, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 120), 1)
    return frame


def draw_landmark_count(frame, count: int):
    cv2.putText(frame, f"Landmarks: {count}/21",
                (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 110, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)


def draw_buffer_bar(frame, buf_len: int):
    """Show how full the sequence buffer is."""
    h, w = frame.shape[:2]
    pct  = buf_len / SEQUENCE_LENGTH
    bw   = int(200 * pct)
    cv2.rectangle(frame, (15, h - 45), (215, h - 30), (40, 40, 40), -1)
    colour = (0, 220, 100) if pct >= 1.0 else (220, 160, 0)
    cv2.rectangle(frame, (15, h - 45), (15 + bw, h - 30), colour, -1)
    cv2.putText(frame, "Buffer", (220, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def main(model_name="mlp"):
    model, scaler, feat_type = load_model(model_name)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # Make window resizable / full-screen
    cv2.namedWindow("Gesture Recognition Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Recognition Demo", 1000, 640)

    sequence       = []                                # rolling feature buffer
    gesture_hist   = collections.deque(maxlen=6)       # last 6 gestures
    last_gesture   = None
    stable_count   = 0
    STABLE_THRESH  = 8                                 # frames before confirming
    screenshot_n   = 0
    prev_time      = time.time()

    print("\n╔══════════════════════════════════════╗")
    print("║   GESTURE RECOGNITION — LIVE DEMO   ║")
    print("╠══════════════════════════════════════╣")
    print("║  Q → quit   S → screenshot   R → reset ║")
    print("╚══════════════════════════════════════╝\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # FPS
        now      = time.time()
        fps      = 1.0 / (now - prev_time + 1e-8)
        prev_time = now

        gesture, conf = None, 0.0

        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            fv = full_feature_vector(hl)
            sequence.append(fv)
            if len(sequence) > SEQUENCE_LENGTH * 2:
                sequence = sequence[-SEQUENCE_LENGTH:]

            gesture, conf = predict(model, scaler, feat_type, sequence)

            # Stability check — only announce when gesture is stable
            if gesture == last_gesture:
                stable_count += 1
            else:
                stable_count  = 0
                last_gesture  = gesture

            if stable_count == STABLE_THRESH and gesture:
                ts = time.strftime("%H:%M:%S")
                gesture_hist.append((ts, gesture))
                trigger_action(gesture)

            # Draw skeleton
            if gesture:
                draw_landmark_skeleton(frame, hl, mp_hands, mp_drawing, gesture)
            draw_landmark_count(frame, len(hl.landmark))
        else:
            # No hand visible
            sequence = []
            stable_count = 0

        # ── Overlays ──────────────────────────────────────────────────────────
        active_gesture = gesture if gesture else "—"
        action = GESTURE_ACTIONS.get(gesture, "Waiting for gesture...") \
                 if gesture else "Waiting for gesture..."

        draw_info_box(frame, active_gesture, conf, action)
        draw_gesture_history(frame, gesture_hist)
        draw_buffer_bar(frame, len(sequence))
        draw_fps(frame, fps)

        # No-hand message
        if not result.multi_hand_landmarks:
            h, w = frame.shape[:2]
            cv2.putText(frame,
                        "Show your hand to the camera",
                        (w//2 - 190, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (100, 100, 100), 2)

        cv2.imshow("Gesture Recognition Demo", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            path = os.path.join(RESULT_DIR, f"screenshot_{screenshot_n:03d}.png")
            cv2.imwrite(path, frame)
            print(f"📸 Screenshot saved → {path}")
            screenshot_n += 1
        elif key == ord('r'):
            gesture_hist.clear()
            sequence = []
            print("🔄 Reset")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp",
                    choices=["mlp", "rf", "svm"],
                    help="Which model to load (default: mlp)")
    args, unknown = ap.parse_known_args()
    main(args.model)


# In[ ]:



