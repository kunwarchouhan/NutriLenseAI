import os
import json
import streamlit as st
from google.cloud import vision, texttospeech
from PIL import Image

# --- Google Cloud Credentials ---
gcp_key = st.secrets["GCP_KEY"]   # already a dict, no need to load
with open("nurti-lens-ai-90a8013ae959.json", "w") as f:
    json.dump(gcp_key, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nurti-lens-ai-90a8013ae959.json"

# --- Initialize Google Clients ---
vision_client = vision.ImageAnnotatorClient()
tts_client = texttospeech.TextToSpeechClient()

# --- Streamlit Page Config ---
st.set_page_config(page_title="Nutrition Lens AI", page_icon="🥗", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00c896;'>🥗 Nutrition Lens AI</h1>", unsafe_allow_html=True)
st.write("Upload a packaged food label to extract **Nutrition Facts**, **Ingredients**, and get a **Health Rating**.")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Browse files", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Extract text using Google Vision
    content = uploaded_file.read()
    image_data = vision.Image(content=content)
    response = vision_client.text_detection(image=image_data)

    texts = response.text_annotations
    if texts:
        extracted_text = texts[0].description
        st.subheader("📜 Extracted Text")
        st.write(extracted_text)

        # --- Simple Health Rating Logic ---
        health_rating = "Healthy"
        if "sugar" in extracted_text.lower():
            health_rating = "Moderate"
        if "trans fat" in extracted_text.lower() or "high sodium" in extracted_text.lower():
            health_rating = "Avoid"

        st.subheader("🩺 Health Rating Result")
        st.write(health_rating)

        # --- Voice Support ---
        if st.button("🔊 Read Nutrition Facts"):
            synthesis_input = texttospeech.SynthesisInput(text=extracted_text)

            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                name="en-IN-Wavenet-A"  # Indian Female Voice
            )

            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

            tts_response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            audio_file = "nutrition.mp3"
            with open(audio_file, "wb") as out:
                out.write(tts_response.audio_content)

            st.audio(audio_file, format="audio/mp3")

    else:
        st.warning("⚠️ Could not extract text. Try another image.")
