
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(page_title="Dog-Cat Classifier", page_icon="🐶")

st.title("🐶 Dog and Cat Image Classification")
st.write("Upload an image to predict whether it is a Dog or Cat.")

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    img = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess image
    img = img.resize((64, 64))

    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, 0)

    # Prediction button
    if st.button("Predict"):

        p = model.predict(x)

        predicted_index = np.argmax(p[0])
        predicted_class = class_names[predicted_index]

        st.success(f"Predicted Class: {predicted_class}")

        st.write("Prediction probabilities:")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {p[0][i]:.4f}")
