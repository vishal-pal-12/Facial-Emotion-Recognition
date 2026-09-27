"""
============================================================
  utils/helpers.py  --  Shared utilities
============================================================
"""

import os
import cv2
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    plt = None
    sns = None

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
# SAFE UNIVERSAL MODEL LOADER (Keras 2 & Keras 3)
# ---------------------------------------------
def safe_load_model(model_path, compile=True):
    """
    Universally loads a Keras model across both Keras 2 and Keras 3 environments.
    Resolves the BatchNormalization axis deserialization incompatibility.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. Try tf_keras (handles legacy Keras 2 models in TensorFlow 2.16+ / Python 3.12)
    try:
        import tf_keras
        return tf_keras.models.load_model(model_path, compile=compile)
    except Exception:
        pass

    # 2. Try standard tf.keras
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path, compile=compile)
    except Exception:
        pass

    # 3. Try standard keras
    try:
        import keras
        return keras.models.load_model(model_path, compile=compile)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_path}'. "
            f"If using TensorFlow >= 2.16 / Python 3.12, please install tf_keras: pip install tf_keras. "
            f"Error: {e}"
        )
