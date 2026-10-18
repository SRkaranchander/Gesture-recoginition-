#!/usr/bin/env python
# coding: utf-8

# In[14]:


#pip install mediapipe opencv-python numpy scikit-learn matplotlib seaborn pyautogui pillow

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import os, sys, time

sys.path.insert(0, os.getcwd())

from utils.gesture_config import (
    GESTURE_LABELS, SEQUENCE_LENGTH,
    full_feature_vector, draw_landmark_skeleton, COLOURS
)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_GESTURE = 50
DATA_DIR            = "data"
OUTPUT_FILE         = os.path.join(DATA_DIR, "gesture_data.npz")
os.makedirs(DATA_DIR, exist_ok=True)

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

def countdown(cap, seconds=3, label=""):
    """Show a live countdown overlay before recording starts."""
    start = time.time()

    while time.time() - start < seconds:
        ret, f = cap.read()
        if not ret:
            continue

        f = cv2.flip(f, 1)
        remaining = seconds - int(time.time() - start)

        cv2.rectangle(f, (0, 0), (f.shape[1], f.shape[0]), (20, 20, 20), -1)
        cv2.putText(f, label, (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 100), 2)

        cv2.putText(f, f"Get ready... {remaining}",
                    (40, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 200, 255), 3)

        cv2.putText(f,
                    "Press SPACE to skip  |  Q to quit",
                    (40, f.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1)

        cv2.imshow("Gesture Data Collection", f)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key == ord(' '):
            return None

    return True


def collect_gesture(cap, gesture_id, gesture_name, sample_idx):
    """Collect one sequence of landmarks."""
    colour   = COLOURS.get(gesture_name, (255, 255, 255))
    sequence = []

    while len(sequence) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        status_colour = (0, 180, 0) if res.multi_hand_landmarks else (0, 0, 200)
        status_text   = "Hand detected" if res.multi_hand_landmarks else "No hand — show hand"

        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            draw_landmark_skeleton(frame, hl, mp_hands, mp_drawing, gesture_name)
            fv = full_feature_vector(hl)
            sequence.append(fv)

        # HUD
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (20, 20, 20), -1)

        cv2.putText(frame,
                    f"Gesture [{gesture_id}]: {gesture_name}",
                    (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    colour, 2)

        cv2.putText(frame,
                    f"Sample {sample_idx+1}/{SAMPLES_PER_GESTURE} | Frame {len(sequence)}/{SEQUENCE_LENGTH}",
                    (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (220, 220, 220), 1)

        cv2.putText(frame, status_text,
                    (15, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    status_colour, 1)

        # Progress bar
        prog = int(frame.shape[1] * len(sequence) / SEQUENCE_LENGTH)
        cv2.rectangle(frame, (0, 118), (prog, 124), colour, -1)

        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord(' '):
            return None

    return np.array(sequence)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Trying camera index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] No camera found.")
        return

    all_X, all_y = [], []

    print("\n══════════════════════════════════════════")
    print("   GESTURE DATA COLLECTION")
    print("══════════════════════════════════════════")

    for gid, gname in GESTURE_LABELS.items():
        print(f"\n[{gid+1}/10] Collecting → {gname}")
        samples_collected = 0

        while samples_collected < SAMPLES_PER_GESTURE:

            result = countdown(cap, seconds=3,
                               label=f"[{gid}] {gname} — sample {samples_collected+1}")

            if result is False:
                break

            if result is None:
                print(f"  Skipped sample {samples_collected+1}")
                samples_collected += 1
                continue

            seq = collect_gesture(cap, gid, gname, samples_collected)

            if seq is None:
                print("  Quit signal received.")
                break

            all_X.append(seq)
            all_y.append(gid)
            samples_collected += 1

            print(f"  ✓ Sample {samples_collected}/{SAMPLES_PER_GESTURE} done")

    if all_X:
        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int32)

        np.savez(OUTPUT_FILE, X=X, y=y)

        print(f"\n✅ Saved {len(all_y)} samples → {OUTPUT_FILE}")
        print(f"   Shape: X={X.shape}  y={y.shape}")
    else:
        print("\n⚠ No data collected.")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# In[ ]:


 

# In[ ]:



