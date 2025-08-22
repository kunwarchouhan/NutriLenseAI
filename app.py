import streamlit as st
from google.cloud import vision
import numpy as np
import cv2
import re


# ---------- Title ----------
st.set_page_config(page_title="AI Nutrition Scanner", layout="centered")
st.title("📦 AI Nutrition Scanner")
st.write("Upload a food package photo and get nutrition facts & ingredients.")

# ---------- Init Google Vision Client ----------
client = vision.ImageAnnotatorClient()

# ---------- OCR Function ----------
def extract_text_google_vision(image_bytes):
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    return ""

# ---------- Nutrition Parser ----------
def parse_nutrition(text):
    nutrition = {}

    # Match values like Protein 5g, Fat 2.3 g, Calories 100 kcal, etc.
    matches = re.findall(r"([A-Za-z]+)\s*[:\-]?\s*([\d\.]+)\s*(mg|g|kcal)?", text, re.IGNORECASE)

    for m in matches:
        key = m[0].strip().capitalize()
        value = m[1].strip()
        unit = m[2].strip() if m[2] else ""
        nutrition[key] = f"{value} {unit}"

    return nutrition

# ---------- Upload ----------
uploaded_file = st.file_uploader("📤 Upload Food Label Image", type=["jpg", "png", "jpeg"])

# ---------- Main Processing ----------
if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # OCR
    text = extract_text_google_vision(file_bytes)

    st.subheader("🔹 Raw Extracted Text")
    st.text_area("", text, height=200)

    # Parse
    nutrition_info = parse_nutrition(text)

    st.subheader("🍎 Parsed Nutrition Info")
    if nutrition_info:
        st.json(nutrition_info)
    else:
        st.warning("⚠️ No nutrition info detected. Try clearer image.")

    # Show image preview
    file_bytes_np = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes_np, 1)
    st.image(img, caption="Uploaded Image", use_column_width=True)
