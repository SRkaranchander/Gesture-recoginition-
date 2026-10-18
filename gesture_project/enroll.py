"""
enroll.py
---------
Face enrollment CLI.
Run this once before starting main.py to register a user's face.

Usage:
    python enroll.py --username alice
"""

import cv2
import sys
import os
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from database import db
from biometric.face_auth import _detect_face, _extract_embedding
from config.settings import ENROLLMENT_SAMPLES
import numpy as np


def enroll(username: str, n_samples: int = ENROLLMENT_SAMPLES) -> None:
    db.initialise()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print(f"\n{'='*50}")
    print(f"  FACE ENROLLMENT — user: {username}")
    print(f"  Capturing {n_samples} samples. Look at the camera.")
    print(f"  Press Q to abort.")
    print(f"{'='*50}\n")

    embeddings = []
    cv2.namedWindow("Enrollment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Enrollment", 640, 480)

    while len(embeddings) < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        face  = _detect_face(frame)

        status_col = (0, 200, 80) if face is not None else (0, 60, 220)
        status_txt = f"Captured {len(embeddings)}/{n_samples}" \
                     if face is not None else "No face detected"

        cv2.putText(frame, status_txt, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2)

        # Progress bar
        pct = len(embeddings) / n_samples
        bw  = int((frame.shape[1] - 40) * pct)
        cv2.rectangle(frame, (20, 60), (frame.shape[1] - 20, 80), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 60), (20 + bw, 80), (0, 200, 80), -1)

        cv2.imshow("Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Enrollment aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return

        if face is not None:
            embeddings.append(_extract_embedding(face))
            time.sleep(0.12)   # slight delay between samples

    cap.release()
    cv2.destroyAllWindows()

    # Average and normalise
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb /= (np.linalg.norm(mean_emb) + 1e-8)

    user_id = db.create_user(username)
    db.save_embedding(user_id, mean_emb.tolist())

    print(f"\n✅ Enrolled '{username}' (user_id={user_id})")
    print(f"   Embedding dim: {len(mean_emb)}")
    print(f"   Stored in: {db.DB_PATH}")
    print("\nYou can now run:  python main.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True, help="Username to enroll")
    ap.add_argument("--samples",  type=int, default=ENROLLMENT_SAMPLES,
                    help=f"Number of face samples (default {ENROLLMENT_SAMPLES})")
    args = ap.parse_args()
    enroll(args.username, args.samples)
