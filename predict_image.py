"""
============================================================
  Facial Emotion Recognition -- Single Image Prediction
  predict_image.py
============================================================
  Usage:
      python predict_image.py --image path/to/face.jpg
      python predict_image.py --image path/to/photo.jpg --no_detect
============================================================
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.helpers import EMOTION_LABELS, CLASS_NAMES, safe_load_model

MODEL_PATH_PRIMARY = 'models/best_model.keras'
MODEL_PATH_FALLBACK = 'models/emotion_cnn_best.keras'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE = 48


def get_model():
    """Load model from primary path or fallback using safe_load_model."""
    if os.path.exists(MODEL_PATH_PRIMARY):
        return safe_load_model(MODEL_PATH_PRIMARY)
    elif os.path.exists(MODEL_PATH_FALLBACK):
        return safe_load_model(MODEL_PATH_FALLBACK)
    else:
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH_PRIMARY}' or '{MODEL_PATH_FALLBACK}'"
        )


def predict_from_image(image_path, detect_face=True, save_result=True, show=False):
    """
    Load an image, optionally detect a face region,
    predict the emotion, and save/display probability results.
    """
    model = get_model()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if detect_face:
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        faces = cascade.detectMultiScale(
            img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            print("[WARN] No face detected with Haar cascade -- using entire image.")
            roi = img_gray
        else:
            x, y, w, h = faces[0]
            roi = img_gray[y:y+h, x:x+w]
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 200, 0), 2)
    else:
        roi = img_gray

    # Preprocess: exact match with training pipeline
    face_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    face_input = face_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    probs = model.predict(face_input, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = EMOTION_LABELS[pred_idx]
    pred_prob = float(probs[pred_idx])

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(
        f"Predicted: {pred_label} ({pred_prob*100:.1f}%)",
        fontsize=14, color='green'
    )
    axes[0].axis('off')

    colors = ['tomato' if i == pred_idx else 'steelblue' for i in range(len(EMOTION_LABELS))]
    bars = axes[1].barh(EMOTION_LABELS, probs * 100, color=colors)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Emotion Probabilities', fontsize=14)
    axes[1].set_xlim(0, 100)
    for bar, p in zip(bars, probs):
        axes[1].text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{p*100:.1f}%', va='center', fontsize=9
        )

    plt.suptitle('Facial Emotion Recognition', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_result:
        os.makedirs('results', exist_ok=True)
        out_path = 'results/prediction_output.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Prediction plot saved -> {out_path}")

    if show:
        plt.show()
    plt.close()

    print(f"\n[RESULT] Emotion    : {pred_label}")
    print(f"         Confidence : {pred_prob*100:.2f}%")
    return pred_label, probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict emotion from image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--no_detect', action='store_true', help='Skip face detection; use whole image')
    parser.add_argument('--show', action='store_true', help='Display the plot window')
    args = parser.parse_args()

    predict_from_image(args.image, detect_face=not args.no_detect, show=args.show)
