# Detection of Covid-19 from Chest X-ray

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-ray Detection",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
.main{
    padding-top:2rem;
}
.stButton>button{
    width:100%;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL PATH
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modell.keras")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_covid_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_covid_model()

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🩺 COVID-19 Chest X-ray Detection")
st.write(
    "Upload a chest X-ray image to predict whether it indicates **COVID-19** or **Normal**."
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("About")
st.sidebar.info(
    """
This application uses a trained Convolutional Neural Network (CNN)
to classify chest X-ray images as:

- COVID-19
- Normal
"""
)

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------
    img = image.resize((299, 299))

    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    with st.spinner("Analyzing X-ray..."):

        prediction = model.predict(img)

        probability = float(prediction[0][0])

        if probability >= 0.5:
            label = "COVID-19"
            confidence = probability
        else:
            label = "Normal"
            confidence = 1 - probability

    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------
    st.success("Prediction Completed")

    if label == "COVID-19":
        st.error(f"### Prediction: {label}")
    else:
        st.success(f"### Prediction: {label}")

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

    st.progress(int(confidence * 100))

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Developed using TensorFlow, Keras & Streamlit</center>",
    unsafe_allow_html=True
)








