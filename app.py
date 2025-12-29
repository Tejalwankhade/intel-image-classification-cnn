import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------
# App Title and Styling
# --------------------------
st.set_page_config(
    page_title="Intel Image Classification",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
        🌍 Intel Image Classification App
    </h1>
    <h4 style='text-align: center;'>
        Developed by <span style="color:#FF4B4B;">Tejal Wankhade</span>
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

st.write("Upload any **natural scene image** and the model will classify it into one of the following:")
st.markdown(
"""
- 🏙 Buildings  
- 🌳 Forest  
- 🧊 Glacier  
- 🏔 Mountain  
- 🌊 Sea  
- 🚗 Street  
"""
)

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("intel_image_model.keras")
    return model

model = load_model()

# --------------------------
# Class labels
# --------------------------
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --------------------------
# File uploader
# --------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # preprocess
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_label = class_names[pred_index]
    confidence = np.max(preds) * 100

    st.markdown("---")

    # Attractive result card
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 15px;
            background-color:#F0F2F6;
            text-align:center;
            border: 2px solid #4B8BBE;
        ">
            <h2>🔍 Prediction Result</h2>
            <h1 style="color:#FF4B4B;">{pred_label.upper()}</h1>
            <h3>✨ Confidence: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("🎉 Classification successful!")

st.markdown("---")
st.caption(" Created by **Tejal Wankhade**")
