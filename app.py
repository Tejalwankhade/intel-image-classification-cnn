import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

st.set_page_config(page_title="Intel Image Classifier", layout="centered")

st.markdown("""
<h1 style='text-align:center; color:#4B8BBE;'>🌍 Intel Natural Scene Image Classifier</h1>
<h4 style='text-align:center;'>Developed by <span style="color:#FF4B4B;">Tejal Wankhade</span></h4>
<hr>
""", unsafe_allow_html=True)

# ---------------- Load TFLite model ----------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="intel_image_model_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.write("🔎 **Model input details:**", input_details)
st.write("🔎 **Model output details:**", output_details)

# ------------ Correct class order ------------
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # ------------- Resize image -------------
    img = image.resize((150, 150))
    img_array = np.array(img)

    # ------------- VERY IMPORTANT: correct preprocessing -------------
    input_dtype = input_details[0]["dtype"]

    # Case 1: quantized model (uint8/int8) → DO NOT scale to 0–1
    if input_dtype == np.uint8 or input_dtype == np.int8:
        st.info("⚙️ Using uint8/int8 quantized preprocessing")
        img_array = img_array.astype(input_dtype)

    # Case 2: float32 model → normalize
    else:
        st.info("⚙️ Using float32 preprocessing")
        img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ------------- Run inference -------------
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])

    st.write("📊 **Raw model output:**", preds)

    pred_index = int(np.argmax(preds))
    pred_label = class_names[pred_index]
    confidence = float(np.max(preds) * 100)

    st.markdown("---")

    st.markdown(f"""
    <div style="background-color:#F8FAFF; padding:20px; border-radius:15px;
                border:2px solid #4B8BBE; text-align:center;">
        <h2>🔍 Prediction</h2>
        <h1 style="color:#FF4B4B;">{pred_label.upper()}</h1>
        <h3>✨ Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.success("🎉 Classification complete!")

st.markdown("---")
st.caption("🚀 Powered by TensorFlow Lite & Streamlit | Crafted with ❤️ by **Tejal Wankhade**")
