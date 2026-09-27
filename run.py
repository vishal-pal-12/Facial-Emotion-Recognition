"""
============================================================
  Facial Emotion Recognition -- Unified Run CLI
  run.py
============================================================
  Commands:
    python run.py train [--epochs 10] [--batch_size 64]
    python run.py evaluate
    python run.py predict --image path/to/image.jpg [--no_detect]
    python run.py webcam [--camera 0]
    python run.py verify
============================================================
"""

import sys
import argparse
import os

# Ensure safe encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def cmd_train(args):
    from train import train_pipeline
    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        from_scratch=args.from_scratch,
        patience=args.patience
    )

def cmd_evaluate(args):
    from train import train_pipeline
    train_pipeline(eval_only=True)

def cmd_predict(args):
    from predict_image import predict_from_image
    predict_from_image(args.image, detect_face=not args.no_detect, show=args.show)

def cmd_webcam(args):
    from realtime_detect import run_realtime
    run_realtime(camera_index=args.camera, max_frames=args.max_frames, output_path=args.save_output)

def cmd_verify(args):
    print("=" * 60)
    print("  RUNNING COMPREHENSIVE END-TO-END PROJECT VERIFICATION")
    print("=" * 60)
    
    # 1. Dataset Verification
    print("\n[1/6] Verifying dataset structure & images ...")
    from utils.helpers import CLASS_NAMES, CLASS_TO_IDX, load_fer2013_folder
    for split in ['train', 'test']:
        p = os.path.join('data/fer2013', split)
        assert os.path.exists(p), f"Missing {p}"
        classes = sorted(os.listdir(p))
        assert classes == sorted(CLASS_NAMES), f"Classes mismatch in {split}: {classes}"
        print(f"  [OK] data/fer2013/{split} contains all 7 classes: {classes}")
    
    # 2. Model file verification
    print("\n[2/6] Verifying trained model files ...")
    model_paths = ['models/best_model.keras', 'models/emotion_cnn_best.keras']
    found_model = None
    for mp in model_paths:
        if os.path.exists(mp):
            print(f"  [OK] Found model at: {mp} ({os.path.getsize(mp) / (1024*1024):.1f} MB)")
            found_model = mp
    assert found_model is not None, "No trained model found in models/"

    # 3. Model loading & prediction check
    print("\n[3/6] Testing model loading & single-image inference ...")
    from utils.helpers import safe_load_model
    model = safe_load_model(found_model)
    test_img_dir = 'data/fer2013/test/happy'
    test_img_file = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])
    from predict_image import predict_from_image
    label, probs = predict_from_image(test_img_file, detect_face=False, save_result=False, show=False)
    print(f"  [OK] Prediction on sample happy image: {label} (confidence: {max(probs)*100:.1f}%)")

    # 4. Evaluation pipeline check on small slice
    print("\n[4/6] Testing evaluation pipeline ...")
    from train import evaluate_model
    evaluate_model(model, test_dir='data/fer2013/test', max_samples=20)
    print("  [OK] Evaluation pipeline passed.")

    # 5. Webcam pipeline simulation check (without requiring physical webcam hardware)
    print("\n[5/6] Testing realtime preprocessing & inference pipeline ...")
    from realtime_detect import preprocess_face, process_frame, CASCADE_PATH
    import cv2
    import numpy as np
    dummy_face = np.ones((100, 100), dtype=np.uint8) * 128
    processed = preprocess_face(dummy_face)
    assert processed.shape == (1, 48, 48, 1), f"Unexpected shape {processed.shape}"
    p_probs = model.predict(processed, verbose=0)[0]
    assert len(p_probs) == 7, "Model output should have 7 probabilities"
    print("  [OK] Realtime preprocessing and inference functions passed.")

    # 6. Results artifacts check
    print("\n[6/6] Checking output artifacts in results/ ...")
    expected_results = [
        'results/confusion_matrix.png',
        'results/training_history.png',
        'results/sample_predictions.png'
    ]
    for r in expected_results:
        exists = os.path.exists(r)
        print(f"  {'[OK]' if exists else '[INFO]'} {r}: {'Found' if exists else 'Not generated yet'}")

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION CHECKS PASSED!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # train
    p_train = subparsers.add_parser("train", help="Train CNN on FER2013")
    p_train.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p_train.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p_train.add_argument("--from_scratch", action="store_true", help="Train from scratch")
    p_train.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate model on test dataset")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict emotion for an image")
    p_pred.add_argument("--image", type=str, required=True, help="Path to image file")
    p_pred.add_argument("--no_detect", action="store_true", help="Skip face detection")
    p_pred.add_argument("--show", action="store_true", help="Display window")

    # webcam
    p_cam = subparsers.add_parser("webcam", help="Run realtime webcam emotion detection")
    p_cam.add_argument("--camera", type=int, default=0, help="Camera index")
    p_cam.add_argument("--max_frames", type=int, default=None, help="Max frames to capture")
    p_cam.add_argument("--save_output", type=str, default=None, help="Save final frame to file")

    # verify
    subparsers.add_parser("verify", help="Run comprehensive verification")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "webcam": cmd_webcam,
        "verify": cmd_verify
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
