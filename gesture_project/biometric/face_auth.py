"""
biometric/face_auth.py
-----------------------
Face-based biometric authentication using OpenCV's built-in face detector
and a lightweight LBP histogram embedding.

No external face_recognition / dlib dependency required.
Raw frames are never stored — only the averaged embedding is persisted.

Public interface
----------------
    enroll_user(cap, username, n_samples) -> (user_id, embedding)
    authenticate(cap, timeout)            -> (user_id | None, confidence)
    verify_frame(frame)                   -> (user_id | None, confidence)
    is_face_present(frame)               -> bool
"""

import cv2
import numpy as np
import os
import sys
import time
import logging
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    BIOMETRIC_CONFIDENCE_THRESHOLD,
    ENROLLMENT_SAMPLES,
)
from database import db

logger = logging.getLogger(__name__)

# ── Haar cascade (ships with OpenCV / local fallback) ─────────────────────────
def _get_cascade_classifier() -> cv2.CascadeClassifier:
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml"),
        getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml" if hasattr(cv2, "data") else "",
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    # Fallback to default path
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, "data") else "haarcascade_frontalface_default.xml")

_face_cascade = _get_cascade_classifier()


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _extract_embedding(gray_face: np.ndarray) -> np.ndarray:
    """
    Compute a normalised LBP histogram as a compact face embedding.
    Returns a 1-D float32 array of length 256.
    """
    face = cv2.resize(gray_face, (64, 64))
    # LBP via uniform comparison
    lbp = np.zeros_like(face, dtype=np.uint8)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(face, dy, axis=0), dx, axis=1)
            lbp += (shifted >= face).astype(np.uint8)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / (norm + 1e-8)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _detect_face(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest detected face as a grayscale crop, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray[y : y + h, x : x + w]


# ── Public interface ───────────────────────────────────────────────────────────

def is_face_present(frame: np.ndarray) -> bool:
    """Return True if at least one face is detected in the frame."""
    return _detect_face(frame) is not None


def enroll_user(
    cap: cv2.VideoCapture,
    username: str,
    n_samples: int = ENROLLMENT_SAMPLES,
) -> tuple[int, list[float]]:
    """
    Capture n_samples face frames, average their embeddings, persist to DB.
    Returns (user_id, embedding_list).
    Raises RuntimeError if not enough faces are captured.
    """
    embeddings: list[np.ndarray] = []
    logger.info("Starting enrollment for '%s' (%d samples)", username, n_samples)

    while len(embeddings) < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        face  = _detect_face(frame)
        if face is not None:
            embeddings.append(_extract_embedding(face))
            logger.debug("Enrollment sample %d/%d", len(embeddings), n_samples)
        time.sleep(0.05)

    if len(embeddings) < n_samples:
        raise RuntimeError("Insufficient face samples captured during enrollment")

    mean_emb = np.mean(embeddings, axis=0)
    mean_emb /= np.linalg.norm(mean_emb) + 1e-8

    user_id = db.create_user(username)
    db.save_embedding(user_id, mean_emb.tolist())
    logger.info("Enrolled user '%s' (id=%d)", username, user_id)
    return user_id, mean_emb.tolist()


def verify_frame(frame: np.ndarray) -> tuple[Optional[int], float]:
    """
    Compare the face in frame against all enrolled embeddings.
    Returns (best_user_id, confidence) or (None, 0.0) if no face / no match.
    """
    profiles = db.load_embeddings()
    if not profiles:
        return None, 0.0

    face = _detect_face(frame)
    if face is None:
        return None, 0.0

    emb = _extract_embedding(face)
    best_uid, best_conf = None, 0.0

    for profile in profiles:
        stored = np.array(profile["embedding"], dtype=np.float32)
        sim    = _cosine_similarity(emb, stored)
        if sim > best_conf:
            best_conf = sim
            best_uid  = profile["user_id"]

    if best_conf >= BIOMETRIC_CONFIDENCE_THRESHOLD:
        return best_uid, best_conf
    return None, best_conf


def authenticate(
    cap: cv2.VideoCapture,
    timeout: float = 10.0,
) -> tuple[Optional[int], float]:
    """
    Attempt face authentication within `timeout` seconds.
    Returns (user_id, confidence) on success, (None, best_conf) on failure.
    Raw frames are discarded after processing.
    """
    deadline = time.time() + timeout
    best_uid, best_conf = None, 0.0

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        uid, conf = verify_frame(frame)
        if conf > best_conf:
            best_conf = conf
            best_uid  = uid
        if uid is not None:
            logger.info("Face authenticated: user=%s conf=%.3f", uid, conf)
            return uid, conf
        time.sleep(0.05)

    logger.warning("Face authentication timed out (best conf=%.3f)", best_conf)
    return None, best_conf


def draw_face_overlay(frame: np.ndarray, uid: Optional[int], conf: float) -> np.ndarray:
    """Draw face detection rectangle and biometric status on frame."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    colour = (0, 200, 80) if uid is not None else (0, 80, 220)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        label = f"ID:{uid} {conf*100:.0f}%" if uid else f"? {conf*100:.0f}%"
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return frame
