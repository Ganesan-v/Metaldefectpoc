# Fix for Windows asyncio bug
import asyncio
import sys
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from utils import generate_report, save_flagged
import os

@st.cache_resource
def load_cnn_model():
    return load_model("model/cnn_model.h5")

model = load_cnn_model()

st.set_page_config(page_title="Metal Defect Detection", layout="wide")
st.title("🔍 Metal Defect Detection System1.0")

uploaded_files = st.file_uploader("Upload image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = {"total": len(uploaded_files), "faulty": 0, "quality_ok": 0}

    for file in uploaded_files:
        try:
            img = Image.open(file).convert("RGB").resize((128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            label = "faulty" if np.argmax(prediction) == 0 else "quality_ok"
            results[label] += 1

            if label == "faulty":
                save_flagged(file)

            st.image(img, caption=f"{file.name} — Prediction: **{label.upper()}**", use_column_width=True)

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    st.subheader("📊 Summary")
    st.write(f"Total Images: {results['total']}")
    st.write(f"Faulty: {results['faulty']}")
    st.write(f"Quality OK: {results['quality_ok']}")

    generate_report(results)
    st.success("✅ Report saved to `reports/analysis_report.txt`")

else:
    st.info("Upload metal images to begin analysis.")
