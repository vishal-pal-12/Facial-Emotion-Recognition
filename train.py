"""
============================================================
  Facial Emotion Recognition using CNN  |  train.py
  Dataset: FER2013 (Folder-based)  |  Framework: TensorFlow / Keras
============================================================
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

from utils.helpers import (
    CLASS_NAMES,
    EMOTION_LABELS,
    CLASS_TO_IDX,
    load_fer2013_folder,
    plot_class_distribution,
    show_sample_images,
    safe_load_model
)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------
IMG_SIZE = 48
BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
NUM_CLASSES = 7
LR = 3e-4

TRAIN_DIR = 'data/fer2013/train'
TEST_DIR = 'data/fer2013/test'
MODEL_PATH = 'models/best_model.keras'
LEGACY_MODEL_PATH = 'models/emotion_cnn_best.keras'
HISTORY_PATH = 'results/training_history.npy'


# ---------------------------------------------
# 1. LOAD & SPLIT FER2013 FROM FOLDERS
# ---------------------------------------------
def load_and_prepare_data(train_dir=TRAIN_DIR, val_split=0.15, max_samples=None):
    """
    Loads training images from folder structure and carves out a stratified validation set.
    Test set is NOT loaded here to guarantee zero data leakage.
    """
    print(f"[INFO] Loading training dataset from '{train_dir}' ...")
    X_full, y_full = load_fer2013_folder(train_dir, img_size=IMG_SIZE, max_samples_per_class=max_samples)
    print(f"[INFO] Total train images loaded: {len(X_full)}")

    # Stratified split: ensures balanced class ratios in both train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=val_split,
        random_state=42,
        stratify=y_full
    )

    print(f"  Training samples   : {len(X_train)} ({len(X_train)/len(X_full)*100:.1f}%)")
    print(f"  Validation samples : {len(X_val)} ({len(X_val)/len(X_full)*100:.1f}%)")
    print("  Train class distribution:")
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_train == idx))
        print(f"    - {name:9s} ({idx}): {count:5d} ({count/len(y_train)*100:.1f}%)")

    return (X_train, y_train), (X_val, y_val)


# ---------------------------------------------
# 2. DATA AUGMENTATION
# ---------------------------------------------
def build_data_generators(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """
    Augment training data only (rotations, shifts, flips, zoom).
    Validation data is never augmented.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen


# ---------------------------------------------
# 3. CNN ARCHITECTURE (Preserved Existing Model)
# ---------------------------------------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES, learning_rate=LR):
    """
    Preserved deep CNN architecture:
      * 4 Conv Blocks (Conv2D -> BN -> Conv2D -> BN -> MaxPool2D -> Dropout(0.25))
      * GlobalAveragePooling2D
      * Dense(512) -> BN -> Dropout(0.5)
      * Dense(256) -> Dropout(0.3)
      * Dense(7, softmax)
      * L2 regularization (1e-4) across Conv and Dense layers
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1 : 64 filters
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2 : 128 filters
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3 : 256 filters
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4 : 512 filters
        layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def initialize_weights(model, source_checkpoint=LEGACY_MODEL_PATH):
    """
    If existing weights are available, transfers them and aligns the output layer
    to the deterministic class mapping:
    [angry(0), disgust(1), fear(2), happy(3), neutral(4), sad(5), surprise(6)].
    """
    if not os.path.exists(source_checkpoint):
        print(f"[INFO] No existing checkpoint found at '{source_checkpoint}', training with random initialization.")
        return model

    try:
        print(f"[INFO] Initializing weights from existing model '{source_checkpoint}' ...")
        source_model = safe_load_model(source_checkpoint)
        for layer in model.layers:
            if layer.name == 'dense_2':
                # Permute classification head:
                # Old CSV mapping:  0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral
                # Target mapping:   0:angry, 1:disgust, 2:fear, 3:happy, 4:neutral, 5:sad, 6:surprise
                perm = [0, 1, 2, 3, 6, 4, 5]
                w, b = source_model.get_layer(layer.name).get_weights()
                layer.set_weights([w[:, perm], b[perm]])
                print(f"  [OK] Aligned '{layer.name}' output weights to deterministic class mapping")
            else:
                try:
                    s_layer = source_model.get_layer(layer.name)
                    if len(s_layer.weights) > 0:
                        layer.set_weights(s_layer.get_weights())
                except Exception:
                    pass
        print("[INFO] Pretrained feature extractor successfully transferred.")
    except Exception as e:
        print(f"[WARN] Could not initialize from checkpoint ({e}), using fresh initialization.")

    return model


# ---------------------------------------------
# 4. CALLBACKS
# ---------------------------------------------
def get_callbacks(patience=6, model_path=MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=max(2, patience // 2),
        min_lr=1e-6,
        verbose=1
    )
    tensorboard = TensorBoard(log_dir='results/logs', histogram_freq=1)

    return [early_stop, checkpoint, reduce_lr, tensorboard]


# ---------------------------------------------
# 5. VISUALISE TRAINING HISTORY
# ---------------------------------------------
def plot_history(history, save_path='results/training_history.png'):
    h = history.history if hasattr(history, 'history') else history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(h['accuracy'], label='Train Accuracy', color='royalblue', lw=2)
    axes[0].plot(h['val_accuracy'], label='Val Accuracy', color='tomato', lw=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(h['loss'], label='Train Loss', color='royalblue', lw=2)
    axes[1].plot(h['val_loss'], label='Val Loss', color='tomato', lw=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training history saved -> {save_path}")


# ---------------------------------------------
# 6. EVALUATE ON TEST SET & METRICS
# ---------------------------------------------
def evaluate_model(model, test_dir=TEST_DIR, max_samples=None):
    """
    Evaluates the model on the untouched test dataset.
    Generates confusion matrix, classification report, precision, recall, F1, accuracy.
    """
    print(f"\n[INFO] Loading untouched test dataset from '{test_dir}' ...")
    X_test, y_test = load_fer2013_folder(test_dir, img_size=IMG_SIZE, max_samples_per_class=max_samples)
    print(f"[INFO] Test dataset size: {len(X_test)} samples")

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n==========================================")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"==========================================")

    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Classification report
    print("\n[INFO] Classification Report:")
    report_text = classification_report(y_test, y_pred, target_names=EMOTION_LABELS, digits=4)
    print(report_text)

    # Precision, recall, f1
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Macro    -> Precision: {p_macro*100:.2f}%, Recall: {r_macro*100:.2f}%, F1: {f1_macro*100:.2f}%")
    print(f"Weighted -> Precision: {p_wt*100:.2f}%, Recall: {r_wt*100:.2f}%, F1: {f1_wt*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS
    )
    plt.title('Confusion Matrix -- Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Emotion Label')
    plt.xlabel('Predicted Emotion Label')
    plt.tight_layout()
    cm_path = 'results/confusion_matrix.png'
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved -> {cm_path}")

    metrics = {
        'test_loss': float(loss),
        'test_accuracy': float(acc),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(p_wt),
        'recall_weighted': float(r_wt),
        'f1_weighted': float(f1_wt),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_text
    }

    return X_test, y_test, metrics


# ---------------------------------------------
# 7. SAMPLE PREDICTIONS VISUALISATION
# ---------------------------------------------
def visualise_predictions(model, X_test, y_test, n=16, save_path='results/sample_predictions.png'):
    idx = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[idx]
    y_true = y_test[idx]
    y_pred = np.argmax(model.predict(X_sample, verbose=0), axis=1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_sample[i].squeeze(), cmap='gray')
        color = 'green' if y_pred[i] == y_true[i] else 'red'
        ax.set_title(
            f"True: {EMOTION_LABELS[y_true[i]]}\nPred: {EMOTION_LABELS[y_pred[i]]}",
            color=color, fontsize=10, fontweight='bold'
        )
        ax.axis('off')
    plt.suptitle('Sample Test Predictions (Green=Correct, Red=Incorrect)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Sample predictions saved -> {save_path}")


# ---------------------------------------------
# 8. MAIN TRAINING PIPELINE
# ---------------------------------------------
def train_pipeline(
    epochs=DEFAULT_EPOCHS,
    batch_size=BATCH_SIZE,
    from_scratch=False,
    eval_only=False,
    patience=5
):
    print("=" * 65)
    print("  Facial Emotion Recognition -- CNN Training Pipeline")
    print("=" * 65)

    if eval_only:
        target_model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
        print(f"[INFO] Evaluation-only mode. Loading '{target_model_path}' ...")
        model = safe_load_model(target_model_path)
        X_test, y_test, metrics = evaluate_model(model, TEST_DIR)
        visualise_predictions(model, X_test, y_test)
        return metrics

    # -- Step 1: Load train data and carve out validation split --
    (X_train, y_train), (X_val, y_val) = load_and_prepare_data(TRAIN_DIR, val_split=0.15)

    # -- Step 2: Compute class weights to handle severe class imbalance --
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print("\n[INFO] Computed balanced class weights:")
    for idx, w in class_weight_dict.items():
        print(f"  {CLASS_NAMES[idx]:9s} : {w:.3f}")

    # -- Step 3: Build augmented generators --
    train_gen, val_gen = build_data_generators(X_train, y_train, X_val, y_val, batch_size=batch_size)

    # -- Step 4: Build & initialize model --
    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES)
    if not from_scratch and os.path.exists(LEGACY_MODEL_PATH):
        model = initialize_weights(model, source_checkpoint=LEGACY_MODEL_PATH)

    model.summary()

    # -- Step 5: Train model --
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    callbacks = get_callbacks(patience=patience, model_path=MODEL_PATH)

    print(f"\n[INFO] Starting training for {epochs} epochs ...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    os.makedirs('results', exist_ok=True)
    np.save(HISTORY_PATH, history.history)
    print(f"[INFO] Training history saved -> {HISTORY_PATH}")

    # Mirror best model to legacy path for compatibility
    if os.path.exists(MODEL_PATH):
        import shutil
        shutil.copyfile(MODEL_PATH, LEGACY_MODEL_PATH)
        print(f"[INFO] Best model also mirrored to -> {LEGACY_MODEL_PATH}")

    # -- Step 6: Plot curves --
    plot_history(history)

    # -- Step 7: Load best saved checkpoint for final evaluation --
    print(f"\n[INFO] Loading best saved model from '{MODEL_PATH}' for final test evaluation ...")
    best_model = safe_load_model(MODEL_PATH)

    # -- Step 8: Evaluate on untouched test dataset --
    X_test, y_test, metrics = evaluate_model(best_model, TEST_DIR)

    # -- Step 9: Sample predictions visualization --
    visualise_predictions(best_model, X_test, y_test)

    # -- Step 10: Class distribution chart --
    plot_class_distribution(y_train, y_val, y_test)

    print("\n" + "=" * 65)
    print(f"  Training and evaluation completed successfully!")
    print(f"  Best Model Path: {MODEL_PATH}")
    print(f"  Final Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print("=" * 65)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN on FER2013')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--from_scratch', action='store_true', help='Train from scratch without transferring weights')
    parser.add_argument('--eval_only', action='store_true', help='Evaluate saved model on test set without training')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        from_scratch=args.from_scratch,
        eval_only=args.eval_only,
        patience=args.patience
    )
