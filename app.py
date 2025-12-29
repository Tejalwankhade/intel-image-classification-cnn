import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Intel Image Classification",
    layout="centered"
)

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4B8BBE;'>
        🌍 Intel Natural Scene Image Classifier
    </h1>

    <h4 style='text-align:center;'>
        Developed by <span style="color:#FF4B4B;">Tejal Wankhade</span>
    </h4>

    <p style='text-align:center; font-size:16px;'>
        Upload an image and the AI model will classify the scene.
    </p>

    <hr>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Load TFLite Model
# ----------------------------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="intel_image_model_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------------------------------
# Class Labels
# ----------------------------------------------------
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ----------------------------------------------------
# Upload Image
# ----------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype("float32")

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])

    pred_index = np.argmax(preds)
    pred_label = class_names[pred_index]
    confidence = np.max(preds) * 100

    st.markdown("---")

    # Fancy result card
    st.markdown(
        f"""
        <div style="
            background-color:#F8FAFF;
            padding:20px;
            border-radius:15px;
            border:2px solid #4B8BBE;
            text-align:center;
        ">
            <h2>🔍 Prediction</h2>
            <h1 style="color:#FF4B4B;">{pred_label.upper()}</h1>
            <h3>✨ Confidence: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("🎉 Image classified successfully!")


st.markdown("---")
st.caption("Created by **Tejal Wankhade**")
