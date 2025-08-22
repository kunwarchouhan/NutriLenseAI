import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re

# ---------- Title ----------
st.set_page_config(page_title="AI Nutrition Scanner", layout="centered")
st.title("📦 AI Nutrition Scanner")
st.write("Upload a food package photo and get nutrition facts & ingredients.")

# ---------- Upload ----------
uploaded_file = st.file_uploader("📤 Upload Food Label Image", type=["jpg", "png", "jpeg"])

# ---------- OCR Function ----------
def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# ---------- Nutrition Parser ----------
def parse_nutrition(text):
    nutrition = {}

    # Match values like Protein 5g, Fat 2.3 g, etc.
    matches = re.findall(r"([A-Za-z]+)\s*[:\-]?\s*([\d\.]+)\s*(mg|g|kcal)?", text, re.IGNORECASE)

    for m in matches:
        key = m[0].strip().capitalize()
        value = m[1].strip()
        unit = m[2].strip() if m[2] else ""
        nutrition[key] = f"{value} {unit}"

    return nutrition

# ---------- Main Processing ----------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # OCR
    text = extract_text(img)

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
    st.image(img, caption="Uploaded Image", use_column_width=True)
