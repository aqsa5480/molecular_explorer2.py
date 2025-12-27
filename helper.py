import os
import tempfile
import base64
import pyttsx3
import streamlit as st

def speak_text(text, voice_type="male", rate="+0%"):
    """
    Offline text-to-speech using pyttsx3, with Streamlit autoplay.
    """

    # Initialize pyttsx3
    engine = pyttsx3.init()

    # Adjust speaking rate
    current_rate = engine.getProperty("rate")
    # rate like "+45%" → parse into integer adjustment
    try:
        if rate.endswith("%"):
            percent = int(rate.strip("%+"))
            if "+" in rate:
                new_rate = current_rate + percent
            elif "-" in rate:
                new_rate = current_rate - percent
            else:
                new_rate = percent
            engine.setProperty("rate", new_rate)
    except Exception:
        pass  # fallback to default rate

    # Set voice (male/female if available)
    voices = engine.getProperty("voices")
    selected_voice = None
    if voice_type == "male":
        for v in voices:
            if "male" in v.name.lower():
                selected_voice = v.id
                break
    elif voice_type == "female":
        for v in voices:
            if "female" in v.name.lower():
                selected_voice = v.id
                break
    if selected_voice:
        engine.setProperty("voice", selected_voice)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        audio_path = tmp_file.name
    engine.save_to_file(text, audio_path)
    engine.runAndWait()

    # Load back and embed in Streamlit
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        b64_audio = base64.b64encode(audio_bytes).decode()

    st.markdown(
        f"""
        <audio autoplay hidden>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )
