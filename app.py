import streamlit as st
from google.cloud import vision, texttospeech
import numpy as np
import cv2
import re
import base64

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="Nutrition Advisor AI", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #d9f4ff, #fef9ff);
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings */
        h1, h2, h3 {
            text-align: center;
            font-weight: 600;
            color: #2c3e50;
        }

        /* Section cards */
        .block-container {
            padding-top: 2rem;
        }
        .stSubheader {
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            font-size: 1.3rem;
        }
        .css-1y4p8pa, .stDataFrame, .stText, .stMarkdown {
            background: rgba(255, 255, 255, 0.7);
            padding: 15px;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }

        /* Buttons */
        button {
            background: linear-gradient(90deg, #0077ff, #00d4ff) !important;
            color: white !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            font-size: 1rem !important;
            font-weight: bold !important;
            border-radius: 30px !important;
            box-shadow: 0 4px 10px rgba(0, 119, 255, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 15px rgba(0, 119, 255, 0.5) !important;
        }

        /* Uploaded Image */
        img {
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        /* Audio Player */
        audio {
            width: 100%;
            margin-top: 10px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("📦 Nutrition Advisor AI")
st.write("✨ Upload a food package photo and get **nutrition facts, ingredients, allergens, health rating & voice playback.**")

# ---------- Init Google Clients ----------
@st.cache_resource
def load_clients():
    vision_client = vision.ImageAnnotatorClient()
    tts_client = texttospeech.TextToSpeechClient()
    return vision_client, tts_client

vision_client, tts_client = load_clients()

# ---------- OCR Function ----------
def extract_text_google_vision(image_bytes):
    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    except Exception as e:
        st.error(f"❌ Vision API Error: {e}")
        return ""

# ---------- Nutrition Parser ----------
def parse_nutrition(text):
    nutrition = {}
    matches = re.findall(r"([A-Za-z\s]+)\s*[:\-]?\s*([\d\.]+)\s*(mg|g|kcal|mcg|%)?", text, re.IGNORECASE)

    for m in matches:
        key = m[0].strip().title()
        value = m[1].strip()
        unit = m[2].strip() if m[2] else ""
        nutrition[key] = f"{value} {unit}".strip()
    return nutrition

# ---------- Ingredients Parser ----------
def parse_ingredients(text):
    ingredients = []
    pattern = re.search(r"(ingredients|contains)[:\-]?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if pattern:
        raw = pattern.group(2)
        raw = re.split(r"\n|\.|;", raw)[0]
        ingredients = [i.strip().capitalize() for i in raw.split(",") if i.strip()]
    return ingredients

# ---------- Allergen Detector ----------
def detect_allergens(ingredients):
    common_allergens = [
        "milk", "egg", "eggs", "peanut", "peanuts", "tree nut", "almond", "cashew",
        "walnut", "hazelnut", "pistachio", "pecan", "brazil nut", "soy", "wheat",
        "gluten", "fish", "shellfish", "shrimp", "crab", "lobster", "sesame"
    ]
    allergens_found = []
    for ing in ingredients:
        for allergen in common_allergens:
            if re.search(rf"\b{allergen}\b", ing, re.IGNORECASE):
                allergens_found.append(allergen.capitalize())
    return sorted(set(allergens_found))

# ---------- Health Rating ----------
def health_rating(nutrition):
    sugar = fat = sodium = 0.0

    for key, val in nutrition.items():
        num = re.findall(r"[\d\.]+", val)
        if not num:
            continue
        try:
            num = float(num[0])
        except ValueError:
            continue

        if "sugar" in key.lower():
            sugar = num
        elif "saturated" in key.lower() or "fat" in key.lower():
            fat = num
        elif "sodium" in key.lower() or "salt" in key.lower():
            sodium = num

    rating = "✅ Healthy"
    color = "success"

    if sugar > 15 or fat > 5 or sodium > 400:
        rating, color = "❌ Avoid", "error"
    elif sugar > 5 or fat > 2 or sodium > 150:
        rating, color = "⚠️ Moderate", "warning"

    return rating, color

# ---------- Voice (TTS) ----------
def speak_text(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN",  # Indian English
        name="en-IN-Wavenet-A"  # Female Indian voice
    )

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    audio_bytes = response.audio_content
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
    <audio autoplay controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# ---------- Image Processing ----------
def process_image(file_bytes):
    with st.spinner("🔍 Extracting text..."):
        text = extract_text_google_vision(file_bytes)

    # Nutrition
    nutrition_info = parse_nutrition(text)
    st.subheader("🍎 Parsed Nutrition Info")
    if nutrition_info:
        st.dataframe(
            [{"Nutrient": k, "Value": v} for k, v in nutrition_info.items()],
            use_container_width=True
        )
        # Voice button
        if st.button("🔊 Read Nutrition Facts"):
            narr_text = "Here are the nutrition facts: " + ", ".join([f"{k}: {v}" for k, v in nutrition_info.items()])
            speak_text(narr_text)
    else:
        st.warning("⚠️ No nutrition info detected. Try uploading a clearer image.")

    # Health Rating
    if nutrition_info:
        rating, color = health_rating(nutrition_info)
        st.subheader("💡 Health Rating")
        getattr(st, color)(f"Overall: **{rating}**")

    # Ingredients
    ingredients = parse_ingredients(text)
    st.subheader("🥗 Ingredients")
    if ingredients:
        st.success(", ".join(ingredients))
    else:
        st.info("No ingredients found in the text.")

    # Allergen Warning
    allergens = detect_allergens(ingredients)
    st.subheader("⚠️ Allergen Warning")
    if allergens:
        st.error("This product may contain: " + ", ".join(allergens))
    else:
        st.info("No common allergens detected.")

    # Preview uploaded image
    file_bytes_np = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_bytes_np, 1)
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

# ---------- Upload ----------
uploaded_file = st.file_uploader("📤 Upload Food Label Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    process_image(file_bytes)
