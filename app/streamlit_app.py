import os
import streamlit as st
import keras
import numpy as np
from PIL import Image
import gdown

# ---------------------------------
# Reduce TensorFlow log noise
# ---------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    layout="centered"
)

# ---------------------------------
# Model Config
# ---------------------------------
MODEL_PATH = "cats_vs_dogs_cnn.keras"

# IMPORTANT: Use this format for gdown
MODEL_URL = "https://drive.google.com/uc?id=1uHxsyYwzjSmsuYMt7H4EHaHT-tSNa2Wc"

# ---------------------------------
# Download model (if not exists)
# ---------------------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("📥 Downloading model... (only first time)")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        # Load model safely
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error("❌ Error loading model")
        st.exception(e)
        return None


model = load_model()

# ---------------------------------
# UI
# ---------------------------------
st.title("🐱🐶 Cats vs Dogs Image Classification")
st.write("Upload an image and the model will predict whether it's a **Cat** or a **Dog**.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# Prediction
# ---------------------------------
if uploaded_file is not None and model is not None:
    try:
        # Display image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)

        # Output
        if prediction[0][0] > 0.5:
            st.success("🐶 Dog")
        else:
            st.success("🐱 Cat")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)
