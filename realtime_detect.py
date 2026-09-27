"""
============================================================
  Facial Emotion Recognition -- Real-Time Webcam Detection
  realtime_detect.py
============================================================
  Uses OpenCV to:
    1. Capture webcam frames (or video/test frame)
    2. Detect faces with Haar cascade
    3. Preprocess face identically to training (48x48, grayscale, /255.0)
    4. Predict emotion with trained CNN
    5. Overlay bounding box, label + confidence, and probability bars
    6. Gracefully handle multiple faces, no faces, and clean exit
============================================================
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.helpers import EMOTION_LABELS, safe_load_model

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
MODEL_PATH_PRIMARY = 'models/best_model.keras'
MODEL_PATH_FALLBACK = 'models/emotion_cnn_best.keras'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE = 48

# Distinct colours per emotion (BGR)
EMOTION_COLORS = {
    'Angry':    (0, 0, 220),       # Red
    'Disgust':  (0, 140, 0),       # Green
    'Fear':     (128, 0, 128),     # Purple
    'Happy':    (0, 215, 255),     # Yellow / Gold
    'Neutral':  (180, 180, 180),   # Light Gray
    'Sad':      (180, 100, 0),     # Dark Blue
    'Surprise': (0, 165, 255),     # Orange
}


def get_model():
    if os.path.exists(MODEL_PATH_PRIMARY):
        return safe_load_model(MODEL_PATH_PRIMARY)
    elif os.path.exists(MODEL_PATH_FALLBACK):
        return safe_load_model(MODEL_PATH_FALLBACK)
    else:
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH_PRIMARY}' or '{MODEL_PATH_FALLBACK}'"
        )


def preprocess_face(face_gray):
    """
    Exact match with training pipeline:
    Resize to 48x48 -> float32 -> normalize to [0, 1] -> shape (1, 48, 48, 1)
    """
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)   # (48, 48, 1)
    face = np.expand_dims(face, axis=0)    # (1, 48, 48, 1)
    return face


def draw_emotion_bars(frame, x, y, w, h, probs):
    """Draw a clean probability bar chart beside or below the face box."""
    bar_x = x + w + 10
    bar_max_w = 120
    bar_h = 14
    bar_gap = 4
    font = cv2.FONT_HERSHEY_SIMPLEX

    # If bar chart would exceed right edge, place it on the left of the face box
    if bar_x + bar_max_w + 50 > frame.shape[1]:
        bar_x = max(10, x - bar_max_w - 60)

    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        bar_y = y + i * (bar_h + bar_gap)
        if bar_y + bar_h > frame.shape[0]:
            break

        color = EMOTION_COLORS.get(label, (200, 200, 200))

        # Background track
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_max_w, bar_y + bar_h),
            (50, 50, 50), -1
        )

        # Filled portion
        filled = int(prob * bar_max_w)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + filled, bar_y + bar_h),
            color, -1
        )

        # Label + percentage text
        cv2.putText(
            frame,
            f"{label[:3]} {prob*100:.0f}%",
            (bar_x + bar_max_w + 4, bar_y + bar_h - 2),
            font, 0.36, (230, 230, 230), 1, cv2.LINE_AA
        )


def process_frame(frame, model, face_cascade):
    """
    Process a single frame:
      - Detect face(s)
      - Preprocess ROI
      - Predict emotion
      - Draw bounding box, label, and probability bars
    Returns annotated frame and list of detections.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detections = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face_input = preprocess_face(roi_gray)

        probs = model.predict(face_input, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = EMOTION_LABELS[pred_idx]
        pred_prob = float(probs[pred_idx])
        color = EMOTION_COLORS.get(pred_label, (0, 255, 0))

        # Face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Header tag
        label_text = f"{pred_label}  {pred_prob*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tag_top = max(0, y - th - 10)
        cv2.rectangle(frame, (x, tag_top), (x + tw + 6, tag_top + th + 10), color, -1)
        cv2.putText(
            frame, label_text, (x + 3, tag_top + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Probability bars
        draw_emotion_bars(frame, x, y, w, h, probs)

        detections.append({
            'box': (x, y, w, h),
            'label': pred_label,
            'confidence': pred_prob,
            'probabilities': probs.tolist()
        })

    # HUD overlay
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 100), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "Press 'q' quit | 's' screenshot",
        (12, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
    )

    return frame, detections


def open_camera(preferred_index=0):
    """Attempt to open camera at preferred index, falling back to 0 or 1."""
    cap = cv2.VideoCapture(preferred_index)
    if cap.isOpened():
        return cap, preferred_index
    cap.release()

    fallback_index = 1 if preferred_index == 0 else 0
    cap = cv2.VideoCapture(fallback_index)
    if cap.isOpened():
        return cap, fallback_index
    cap.release()
    return None, None


def run_realtime(camera_index=0, max_frames=None, output_path=None):
    """
    Main webcam detection loop.
    Supports max_frames limit for automated testing.
    """
    print("[INFO] Loading emotion model ...")
    model = get_model()
    print("[INFO] Model loaded successfully.")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {CASCADE_PATH}")

    cap, actual_index = open_camera(camera_index)
    if cap is None:
        print(f"[ERROR] Could not open webcam at index {camera_index} or fallback.")
        print("[INFO] Tip: Verify physical camera connection and permissions.")
        return False

    print(f"[INFO] Webcam started on device index {actual_index}.")
    print("[INFO] Press 'q' to quit, 's' to save screenshot.")
    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed or camera disconnected.")
                break

            annotated_frame, detections = process_frame(frame, model, face_cascade)
            cv2.imshow('Facial Emotion Recognition -- Real-Time', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                os.makedirs('results', exist_ok=True)
                screenshot_path = f'results/screenshot_{saved_count:04d}.png'
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"[INFO] Screenshot saved -> {screenshot_path}")
                saved_count += 1

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                if output_path:
                    cv2.imwrite(output_path, annotated_frame)
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released cleanly.")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time webcam emotion recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--max_frames', type=int, default=None, help='Max frames to capture (for testing)')
    parser.add_argument('--save_output', type=str, default=None, help='Save final frame to file')
    args = parser.parse_args()

    run_realtime(camera_index=args.camera, max_frames=args.max_frames, output_path=args.save_output)
