"""Streamlit web application for Image Sentiment Analysis.

Run with::

    streamlit run app.py
"""

import os
from pathlib import Path
from typing import List

import streamlit as st
import torch
from PIL import Image

from src.config import DATA_CONFIG, MODEL_CONFIG
from src.inference.predict import load_model, predict
from src.models.model import build_model

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Sentiment Analysis",
    page_icon="🎭",
    layout="centered",
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "models/best_model.pth"
DEFAULT_IMG_SIZE = DATA_CONFIG["img_size"]
BACKBONE_OPTIONS: List[str] = ["custom_cnn", "resnet18", "efficientnet_b0"]

SENTIMENT_EMOJI = {
    "happy": "😊",
    "sad": "😞",
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
}


# ── Model loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_cached_model(checkpoint_path: str, backbone: str):
    """Cache the model so it is only loaded once per session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        backbone=backbone,
        num_classes=2,
        dropout=MODEL_CONFIG["dropout"],
    )
    model, class_names = load_model(model, checkpoint_path, device)
    return model, class_names, device


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    checkpoint_path = st.text_input(
        "Checkpoint path", value=DEFAULT_CHECKPOINT
    )
    backbone = st.selectbox("Model backbone", BACKBONE_OPTIONS)
    img_size = st.slider("Image size", 64, 512, DEFAULT_IMG_SIZE, step=32)

    st.markdown("---")
    st.markdown(
        "**How to train a model:**\n"
        "```bash\n"
        "python train.py\n"
        "```"
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎭 Image Sentiment Analysis")
st.markdown(
    "Upload an image to classify its sentiment (e.g. **Happy** vs **Sad**) "
    "using a trained Convolutional Neural Network."
)

# Check for checkpoint
checkpoint_exists = os.path.isfile(checkpoint_path)
if not checkpoint_exists:
    st.warning(
        f"⚠️ No checkpoint found at **{checkpoint_path}**. "
        "Please train a model first (`python train.py`) or provide "
        "the path to an existing `.pth` checkpoint in the sidebar."
    )

uploaded_file = st.file_uploader(
    "Choose an image …",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    disabled=not checkpoint_exists,
)

if uploaded_file is not None and checkpoint_exists:
    col_img, col_result = st.columns([1, 1])

    with col_img:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference …"):
        try:
            model, class_names, device = load_cached_model(checkpoint_path, backbone)

            # Save upload to a temp path so predict() can open it
            import tempfile

            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            result = predict(
                str(tmp_path), model, class_names, img_size, device
            )
        except Exception as exc:
            st.error(f"Inference error: {exc}")
            st.stop()

    with col_result:
        predicted_class = result["class"]
        emoji = SENTIMENT_EMOJI.get(predicted_class.lower(), "🔍")

        st.markdown(f"## {emoji} {predicted_class.capitalize()}")
        st.metric("Confidence", f"{result['confidence']:.1%}")

        st.markdown("**Class Probabilities**")
        for cls, prob in result["probabilities"].items():
            e = SENTIMENT_EMOJI.get(cls.lower(), "")
            st.progress(prob, text=f"{e} {cls.capitalize()}: {prob:.1%}")

elif uploaded_file is not None:
    st.info("Please provide a valid checkpoint path in the sidebar first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Image Sentiment Analysis · Built with PyTorch & Streamlit · "
    "[GitHub](https://github.com/Rishabhh3/Image-Sentiment-Analysis)"
)
