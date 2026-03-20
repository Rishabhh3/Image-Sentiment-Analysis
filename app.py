import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.models.model import SentimentModel, get_device
from src.inference.predict import SentimentPredictor


st.set_page_config(
	page_title="Image Sentiment Analysis",
	page_icon="🙂",
	layout="wide",
)


def _discover_model_candidates():
	preferred = [
		Path("models/sentiment_model.pth"),
		Path("sentiment_model.pth"),
		Path("model.pth"),
	]

	# Keep preferred names first if they exist.
	ordered = [p for p in preferred if p.exists()]

	# Add any other discovered .pth files without duplicates.
	for p in sorted(Path(".").rglob("*.pth")):
		if p not in ordered:
			ordered.append(p)

	return ordered


def _infer_class_names_from_data(data_dir: Path):
	if not data_dir.exists() or not data_dir.is_dir():
		return ["happy", "sad"]
	class_dirs = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
	return class_dirs if class_dirs else ["happy", "sad"]


@st.cache_resource
def load_predictor(model_path: str):
	path = Path(model_path)
	if not path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	class_names = _infer_class_names_from_data(Path("Data"))
	device = get_device()
	model = SentimentModel(num_classes=len(class_names), pretrained=False)
	model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
	predictor = SentimentPredictor(model=model, device=device, sentiment_labels=class_names)
	return predictor, class_names, str(device)


def main():
	st.title("Image Sentiment Analysis")
	st.caption("Upload an image and predict whether the sentiment is happy or sad.")
	candidates = _discover_model_candidates()

	with st.sidebar:
		st.header("Settings")
		default_model = str(candidates[0]) if candidates else "models/sentiment_model.pth"
		model_path = st.text_input("Model path", value=default_model)
		if candidates:
			st.caption("Detected model files")
			for p in candidates:
				st.write(f"- {p}")
		st.markdown("Train your model first with: `python main.py --mode train`")

	col_left, col_right = st.columns([1, 1])

	with col_left:
		if not candidates and not Path(model_path).exists():
			st.warning("No model file was found in this project.")
			st.info("Run training first: python main.py --mode train")

		uploaded = st.file_uploader(
			"Upload an image",
			type=["jpg", "jpeg", "png", "bmp"],
			accept_multiple_files=False,
		)

		if uploaded is not None:
			image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
			st.image(image, caption="Uploaded image", use_container_width=True)

			if st.button("Predict sentiment", type="primary"):
				try:
					if not Path(model_path).exists():
						st.error(f"Model file not found at: {model_path}")
						st.info("Tip: train first with python main.py --mode train or set the correct file path in the sidebar.")
						return

					predictor, class_names, device_name = load_predictor(model_path)
					result = predictor.predict(io.BytesIO(uploaded.getvalue()))

					if "error" in result:
						st.error(result["error"])
						return

					st.success("Prediction complete")
					st.write(f"Device: {device_name}")
					st.write(f"Class mapping: {class_names}")

					sentiment = result["sentiment"]
					confidence = result["confidence"]
					st.metric("Predicted sentiment", sentiment.upper())
					st.metric("Confidence", f"{confidence * 100:.2f}%")

					st.subheader("Class probabilities")
					probs = result["probabilities"]
					for label, prob in probs.items():
						st.progress(float(prob), text=f"{label}: {prob * 100:.2f}%")

				except Exception as e:
					st.error(str(e))

	with col_right:
		st.subheader("How to use")
		st.markdown(
			"""
			1. Train a model and keep weights in `models/sentiment_model.pth`
			2. Start this app with `streamlit run app.py`
			3. Upload an image and click **Predict sentiment**
			"""
		)


if __name__ == "__main__":
	main()
