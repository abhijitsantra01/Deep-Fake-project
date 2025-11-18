import streamlit as st
import numpy as np
import cv2
import tempfile
import tensorflow as tf

st.set_page_config(page_title="Deepfake Detection App", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake.h5")

model = load_model()

# -------------------------------------------------
# Extract raw video frames (same as your training)
# -------------------------------------------------
def extract_raw_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # sample evenly spaced frames
    frame_ids = np.linspace(0, total - 1, max_frames, dtype=int)

    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (128,128))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()
    return np.array(frames)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🔍 Deepfake Video Detection")
uploaded_file = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])

if uploaded_file:
    st.video(uploaded_file)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_file.read())
    tmp.flush()

    st.info("Extracting frames (no face detection, raw frames)...")

    frames = extract_raw_frames(tmp.name)

    if len(frames) == 0:
        st.error("Unable to extract frames.")
    else:
        st.success(f"{len(frames)} frames extracted!")

        preds = []
        for f in frames:
            p = model.predict(np.expand_dims(f, 0))[0]
            preds.append(p)

        preds = np.array(preds)
        avg_pred = preds.mean(axis=0)

        classes = ["REAL", "FAKE"]
        label = classes[np.argmax(avg_pred)]
        confidence = avg_pred[np.argmax(avg_pred)]

        st.subheader("🧠 Prediction")
        if label == "FAKE":
            st.error(f"❌ FAKE — confidence: {confidence:.4f}")
        else:
            st.success(f"✅ REAL — confidence: {confidence:.4f}")
