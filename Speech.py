import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import speech_recognition as sr
import whisper
import numpy as np
import torch
from streamlit_js_eval import streamlit_js_eval

@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("base")
    return model

def transcribe_audio():
    r = sr.Recognizer()
    model = load_whisper_model()

    with sr.Microphone(sample_rate=16000) as source:
        status_placeholder.info("Adjusting for background noise...", icon ="⏳")
        r.adjust_for_ambient_noise(source, duration= 1)
        status_placeholder.info("Listening...", icon ="🎤")

        try:
            audio = r.record(source, duration=10)
            raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

            if np.max(np.abs(raw_data)) < 0.01:
                st.warning("No significant audio detected. Please try again.")
                return None
            
            data = torch.from_numpy(raw_data)
            status_placeholder.info("Processing speech with Whisper...", icon ="⏳")
            result = model.transcribe(data, fp16=False)
            return result["text"]

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

st.title("Speech Recognition with Whisper")
st.write("Click the button below and speak into your microphone. The transcribed text will be displayed")
st.write(" **Recording automatically ends after 10 seconds**")

if "transcript_history" not in st.session_state:
    st.session_state.transcript_history = []

status_placeholder = st.empty()

if st.button("Start Recording", type="primary"):
    text = transcribe_audio()
    if text:
        st.session_state.transcript_history.append(text)
        status_placeholder.success("Transcription complete!", icon ="✅")
    else:
        status_placeholder.empty()

st.divider()
st.subheader("Transcription History")
for i, entry in enumerate(reversed(st.session_state.transcript_history)):
    st.text_area(f"Capture {len(st.session_state.transcript_history)- i}", value = entry, height = 100)

if st.button("clear history"):
    st.session_state.transcript_history = []
    st.rerun()