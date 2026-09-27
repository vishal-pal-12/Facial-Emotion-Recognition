"""
============================================================
  utils/helpers.py  --  Shared utilities
============================================================
"""

import os
import cv2
import numpy as np

# Deterministic and explicit class-index mapping
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------
# DATASET LOADING FROM FOLDERS
# ---------------------------------------------
def load_fer2013_folder(folder_path, img_size=48, max_samples_per_class=None):
    """
    Load FER2013 images from directory containing class subfolders:
    angry, disgust, fear, happy, neutral, sad, surprise.
    Returns:
        X: float32 normalized to [0, 1], shape (N, img_size, img_size, 1)
        y: int32 class indices [0..6], shape (N,)
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    X_list, y_list = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(folder_path, cls_name)
        if not os.path.exists(cls_dir):
            raise FileNotFoundError(f"Missing expected class folder: {cls_dir}")
        fnames = sorted(os.listdir(cls_dir))
        if max_samples_per_class is not None:
            fnames = fnames[:max_samples_per_class]

        for fname in fnames:
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (img_size, img_size):
                img = cv2.resize(img, (img_size, img_size))
            X_list.append(img)
            y_list.append(cls_idx)

    X = np.array(X_list, dtype=np.float32).reshape(-1, img_size, img_size, 1) / 255.0
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ---------------------------------------------
# DATASET STATS
# ---------------------------------------------
def plot_class_distribution(y_train, y_val=None, y_test=None, save_path='results/class_distribution.png'):
    """
    Bar chart showing sample counts per emotion class in each split.
    Highlights the class imbalance in FER2013 (Disgust is rare).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(EMOTION_LABELS))
    w = 0.25

    counts_train = [np.sum(y_train == i) for i in range(len(EMOTION_LABELS))]
    ax.bar(x - w, counts_train, width=w, label='Train', color='royalblue', alpha=0.85)

    if y_val is not None:
        counts_val = [np.sum(y_val == i) for i in range(len(EMOTION_LABELS))]
        ax.bar(x, counts_val, width=w, label='Validation', color='orange', alpha=0.85)

    if y_test is not None:
        counts_test = [np.sum(y_test == i) for i in range(len(EMOTION_LABELS))]
        ax.bar(x + w, counts_test, width=w, label='Test', color='green', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=20)
    ax.set_ylabel('Sample Count')
    ax.set_title('Class Distribution per Split')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Class distribution saved -> {save_path}")


# ---------------------------------------------
# VISUALISE SAMPLE IMAGES
# ---------------------------------------------
def show_sample_images(X, y, n_per_class=4, save_path='results/sample_images_per_class.png'):
    """Show n_per_class sample images for each emotion label."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(EMOTION_LABELS), n_per_class,
                             figsize=(n_per_class * 2, len(EMOTION_LABELS) * 2))
    for cls_idx, label in enumerate(EMOTION_LABELS):
        indices = np.where(y == cls_idx)[0]
        if len(indices) == 0:
            continue
        chosen = np.random.choice(indices, min(n_per_class, len(indices)), replace=False)
        for j, idx in enumerate(chosen):
            axes[cls_idx][j].imshow(X[idx].squeeze(), cmap='gray')
            axes[cls_idx][j].axis('off')
            if j == 0:
                axes[cls_idx][j].set_ylabel(label, fontsize=10, rotation=0,
                                             labelpad=45, va='center')
    plt.suptitle('Sample Images per Emotion Class', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Sample images saved -> {save_path}")


# ---------------------------------------------
# GRAD-CAM (simple version for last conv layer)
# ---------------------------------------------
def grad_cam(model, img_array, layer_name=None):
    """
    Compute Grad-CAM heatmap for a single preprocessed image (1,48,48,1).
    If layer_name is None, uses the last Conv2D layer automatically.
    Returns the heatmap (48x48 float32).
    """
    import tensorflow as tf

    if layer_name is None:
        # Find last Conv2D layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img_gray, heatmap):
    """Overlay Grad-CAM heatmap on the original grayscale face image."""
    img_rgb = cv2.cvtColor(
        (original_img_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


# ---------------------------------------------
# SAFE UNIVERSAL MODEL LOADER (H5 & Keras 2/3)
# ---------------------------------------------
def safe_load_model(model_path=None, compile=False):
    """
    Universally loads a Keras model across both Keras 2 and Keras 3 environments.
    Checks H5 first (ultra-compatible & fast), then .keras with tf.keras, tf_keras, and keras.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    search_paths = []
    if model_path:
        search_paths.append(model_path)
        if not os.path.isabs(model_path):
            search_paths.append(os.path.join(root_dir, model_path))
        if model_path.endswith('.keras'):
            search_paths.insert(0, model_path.replace('.keras', '.h5'))
            search_paths.insert(1, os.path.join(root_dir, model_path.replace('.keras', '.h5')))

    for standard_name in ['models/best_model.h5', 'models/best_model.keras', 'models/emotion_cnn_best.keras']:
        search_paths.append(standard_name)
        search_paths.append(os.path.join(root_dir, standard_name))

    seen = set()
    valid_candidates = []
    for p in search_paths:
        abs_p = os.path.abspath(p)
        if abs_p not in seen and os.path.exists(abs_p):
            seen.add(abs_p)
            valid_candidates.append(abs_p)

    if not valid_candidates:
        raise FileNotFoundError(
            f"No model file found. Checked: {search_paths}. "
            f"Working dir: {os.getcwd()}, Root dir: {root_dir}"
        )

    # Put .h5 models first (immune to Keras 3 BatchNormalization axis deserialization bug)
    h5_candidates = [p for p in valid_candidates if p.endswith('.h5')]
    other_candidates = [p for p in valid_candidates if not p.endswith('.h5')]
    sorted_candidates = h5_candidates + other_candidates

    errors = []
    for path in sorted_candidates:
        # 1. Standard tf.keras (handles .h5 flawlessly with compile=False)
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(path, compile=compile)
        except Exception as e_tf:
            errors.append(f"tf.keras on {os.path.basename(path)}: {e_tf}")

        # 2. tf_keras (handles legacy Keras 2 models in TF 2.16+)
        try:
            import tf_keras
            return tf_keras.models.load_model(path, compile=compile)
        except Exception as e_tfk:
            errors.append(f"tf_keras on {os.path.basename(path)}: {e_tfk}")

        # 3. Native keras
        try:
            import keras
            return keras.models.load_model(path, compile=compile)
        except Exception as e_k:
            errors.append(f"keras on {os.path.basename(path)}: {e_k}")

    raise RuntimeError(
        f"Failed to load model from candidates: {[os.path.basename(p) for p in sorted_candidates]}. "
        f"Details: {'; '.join(errors)}"
    )
