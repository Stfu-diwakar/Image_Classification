import os
import urllib.request
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -----------------------------
# Model download from Google Drive
# -----------------------------
MODEL_PATH = "model/cats_vs_dogs_cnn.h5"
MODEL_URL = "https://drive.google.com/uc?id=1LrLSyQtccj1PYYMMhfwcygmiVHgTp5XC"

# Create model directory if not exists
if not os.path.exists("model"):
    os.makedirs("model")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Cats vs Dogs Classifier", layout="centered")

st.title("🐱🐶 Cats vs Dogs Image Classification")
st.write("Upload an image and the CNN model will predict whether it is a **Cat** or a **Dog**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Prediction: 🐶 Dog")
    else:
        st.success("Prediction: 🐱 Cat")
