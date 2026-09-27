"""
============================================================
  Facial Emotion Recognition — Streamlit Cloud Web Application
  app.py
  Author: Vishal Pal (github.com/vishal-pal-12)
============================================================
"""

import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Facial Emotion Recognition AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CONSTANTS & CONFIG
# -------------------------------------------------------------
MODEL_PATH = "models/best_model.h5"
FALLBACK_MODEL_PATH = "models/best_model.keras"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = 48

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}
EMOTION_COLORS = {
    "Angry": "#FF4B4B",
    "Disgust": "#2ECC71",
    "Fear": "#9B59B6",
    "Happy": "#F1C40F",
    "Neutral": "#95A5A6",
    "Sad": "#3498DB",
    "Surprise": "#E67E22"
}
EMOTION_COLORS_BGR = {
    "Angry": (75, 75, 255),       # Red
    "Disgust": (113, 204, 46),    # Green
    "Fear": (182, 89, 155),       # Purple
    "Happy": (15, 196, 241),      # Gold / Yellow
    "Neutral": (166, 165, 149),   # Gray
    "Sad": (219, 152, 52),        # Blue
    "Surprise": (34, 126, 230)    # Orange
}

# WebRTC Streaming Setup (for continuous real-time webcam)
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:global.stun.twilio.com:3478"]}
            ]
        }
    )
except Exception:
    WEBRTC_AVAILABLE = False
    RTC_CONFIGURATION = None

# -------------------------------------------------------------
# CUSTOM STYLING (CSS)
# -------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #6C5CE7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        background-color: rgba(108, 92, 231, 0.2);
        border: 1px solid rgba(108, 92, 231, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# MODEL LOADER (CACHED)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_emotion_model():
    """Loads model universally using safe_load_model with compile=False."""
    from utils.helpers import safe_load_model
    try:
        return safe_load_model(MODEL_PATH, compile=False)
    except Exception as err:
        st.error(f"Error loading emotion recognition model: {err}")
        return None


model = load_emotion_model()


# -------------------------------------------------------------
# FACE DETECTION & PREPROCESSING
# -------------------------------------------------------------
def detect_and_preprocess_faces(image_np):
    """
    Detects faces using Haar Cascade.
    Returns:
        annotated_image: RGB image with drawn bounding boxes & labels
        face_crops: list of (48, 48, 1) normalized tensors
        boxes: list of (x, y, w, h)
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    annotated = image_np.copy()
    face_crops = []
    boxes = []

    if len(faces) == 0:
        # If no face detected, crop central region or use full image
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=(0, -1))
        return annotated, [tensor], [(0, 0, image_np.shape[1], image_np.shape[0])], False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=(0, -1))
        face_crops.append(tensor)
        boxes.append((x, y, w, h))

    return annotated, face_crops, boxes, True


# -------------------------------------------------------------
# WEBRTC VIDEO PROCESSOR (CONTINUOUS REAL-TIME DETECTION)
# -------------------------------------------------------------
if WEBRTC_AVAILABLE:
    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.cascade = cv2.CascadeClassifier(CASCADE_PATH)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
                tensor = resized.astype(np.float32) / 255.0
                tensor = np.expand_dims(tensor, axis=(0, -1))

                if model is not None:
                    # Ultra-fast direct graph call (~15-30ms)
                    preds = model(tensor, training=False).numpy()[0]
                    pred_idx = int(np.argmax(preds))
                    pred_label = EMOTIONS[pred_idx]
                    confidence = float(preds[pred_idx]) * 100
                    color_bgr = EMOTION_COLORS_BGR.get(pred_label, (46, 204, 113))

                    # Draw bounding box
                    cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 3)

                    # Text Header Label with solid background
                    label_text = f"{pred_label} {confidence:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x, max(0, y - th - 12)), (x + tw + 10, y), color_bgr, -1)
                    cv2.putText(
                        img,
                        label_text,
                        (x + 5, max(th + 2, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=90)
    st.title("Emotion AI System")
    st.caption("Deep Learning Facial Emotion Recognition")

    st.markdown("---")
    st.markdown("### 📊 Model Architecture & Stats")
    st.markdown("""
    - **Architecture:** 4-Block Deep CNN
    - **Parameters:** `5,089,735`
    - **Input Shape:** `(48, 48, 1)` Grayscale
    - **Classes:** `7` Emotions
    - **Test Accuracy:** **`65.38%`**
    - **Best Val Accuracy:** **`69.40%`**
    - **Weighted F1 Score:** **`65.05%`**
    """)

    st.markdown("---")
    st.markdown("### 🏷️ 7 Emotion Classes")
    for emo, emoji in EMOTION_EMOJIS.items():
        st.markdown(f"- **{emoji} {emo}**")

    st.markdown("---")
    with st.expander("📈 View Confusion Matrix & Training History"):
        if os.path.exists("results/confusion_matrix.png"):
            st.image("results/confusion_matrix.png", caption="Test Confusion Matrix (7,178 images)")
        if os.path.exists("results/training_history.png"):
            st.image("results/training_history.png", caption="Training & Validation Curves")

    st.markdown("---")
    st.markdown("**Author:** [Vishal Pal](https://github.com/vishal-pal-12)")
    st.caption("Facial Emotion Recognition Project")


# -------------------------------------------------------------
# MAIN VIEW
# -------------------------------------------------------------
st.markdown('<div class="main-header">🎭 Facial Emotion Recognition AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time facial expression analysis powered by Deep Convolutional Neural Networks on FER-2013</div>', unsafe_allow_html=True)

# Mode Selector
mode = st.radio(
    "Choose Input Source:",
    [
        "🎥 Continuous Live Stream (Real-Time)",
        "📸 Snapshot Camera (Photo Booth)",
        "📁 Upload Image File",
        "🖼️ Sample Test Gallery"
    ],
    horizontal=True
)

st.markdown("---")

input_image = None

# Mode 1: Continuous Live Stream (Real-Time WebRTC)
if mode == "🎥 Continuous Live Stream (Real-Time)":
    st.markdown("#### 🎥 Live Camera Stream (Real-Time Emotion Tracking)")
    st.info("Click **'START'** below to activate your camera. The AI model will track your face and predict emotions continuously frame-by-frame in real-time, exactly like the desktop app!")

    if WEBRTC_AVAILABLE:
        col_stream, col_info = st.columns([1.6, 1], gap="large")
        with col_stream:
            webrtc_streamer(
                key="emotion-live-stream",
                video_processor_factory=EmotionVideoProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        with col_info:
            st.markdown("##### 💡 How to use")
            st.markdown("""
            - Click **START** to turn on the camera.
            - Allow webcam access when prompted by the browser.
            - Look straight into the camera.
            - Change facial expressions (**Happy, Angry, Surprise, Sad, etc.**).
            - The bounding box and emotion label track your face automatically.
            - Click **STOP** when you want to pause or turn off the camera.
            """)
            st.markdown("##### 🏷️ 7 Recognizable Emotions")
            for emo, emoji in EMOTION_EMOJIS.items():
                st.markdown(f"- **{emoji} {emo}**")
    else:
        st.warning("`streamlit-webrtc` is not available in the current environment.")

# Mode 2: Snapshot Camera (Photo Booth)
elif mode == "📸 Snapshot Camera (Photo Booth)":
    st.write("Take a snapshot with your device camera to predict emotions and view full 7-class probability breakdown:")
    camera_photo = st.camera_input("Smile / Express an Emotion")
    if camera_photo is not None:
        input_image = Image.open(camera_photo)

# Mode 2: Upload File
elif mode == "📁 Upload Image File":
    uploaded_file = st.file_uploader("Upload a face image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

# Mode 3: Sample Gallery
elif mode == "🖼️ Sample Test Gallery":
    st.write("Select a pre-verified test image from the FER-2013 dataset:")
    sample_files = {
        "Happy 😄": "sample_images/happy.jpg",
        "Surprise 😲": "sample_images/surprise.jpg",
        "Sad 😢": "sample_images/sad.jpg",
        "Neutral 😐": "sample_images/neutral.jpg",
        "Angry 😠": "sample_images/angry.jpg",
        "Fear 😨": "sample_images/fear.jpg",
        "Disgust 🤢": "sample_images/disgust.jpg"
    }

    selected_sample = st.selectbox("Choose sample expression:", list(sample_files.keys()))
    sample_path = sample_files[selected_sample]
    if os.path.exists(sample_path):
        input_image = Image.open(sample_path)
    else:
        st.warning(f"Sample file '{sample_path}' not found.")


# -------------------------------------------------------------
# INFERENCE & RESULTS DISPLAY
# -------------------------------------------------------------
if input_image is not None and model is not None:
    # Convert PIL Image to RGB Numpy array
    image_np = np.array(input_image.convert("RGB"))

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Analyzed Face")
        with st.spinner("Detecting faces and classifying emotion..."):
            annotated_img, face_crops, boxes, face_found = detect_and_preprocess_faces(image_np)

            all_predictions = []
            for i, crop in enumerate(face_crops):
                probs = model.predict(crop, verbose=0)[0]
                pred_idx = np.argmax(probs)
                pred_label = EMOTIONS[pred_idx]
                confidence = probs[pred_idx] * 100
                all_predictions.append((pred_label, confidence, probs))

                if face_found:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (46, 204, 113), 3)
                    cv2.putText(
                        annotated_img,
                        f"{pred_label} ({confidence:.1f}%)",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (46, 204, 113),
                        2
                    )

            st.image(annotated_img, use_container_width=True)
            if not face_found:
                st.info("ℹ️ No face detected with Haar Cascade. Analyzed the full image frame.")

    with col2:
        st.subheader("🎯 Emotion Classification")

        if len(all_predictions) > 0:
            top_label, top_conf, top_probs = all_predictions[0]
            top_emoji = EMOTION_EMOJIS[top_label]

            # Big metric display
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.05); border: 2px solid {EMOTION_COLORS[top_label]}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 3.5rem; margin-bottom: 5px;">{top_emoji}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {EMOTION_COLORS[top_label]};">{top_label.upper()}</div>
                    <div style="font-size: 1.2rem; color: #BBBBBB;">Confidence: <b>{top_conf:.2f}%</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("#### 📊 Probability Distribution:")
            for emo, prob in zip(EMOTIONS, top_probs):
                col_name, col_bar = st.columns([1, 3])
                with col_name:
                    st.write(f"{EMOTION_EMOJIS[emo]} **{emo}**")
                with col_bar:
                    pct = float(prob)
                    st.progress(pct, text=f"{pct*100:.1f}%")

            # Multi-face alert if multiple faces detected
            if len(all_predictions) > 1:
                st.markdown("---")
                st.write(f"👥 **{len(all_predictions)} faces detected in image:**")
                for idx, (lbl, conf, _) in enumerate(all_predictions):
                    st.write(f"- Face #{idx+1}: **{EMOTION_EMOJIS[lbl]} {lbl}** ({conf:.1f}%)")

else:
    # Initial state greeting
    st.info("👋 Select an input method above (Webcam, Upload, or Sample) to begin emotion recognition.")
