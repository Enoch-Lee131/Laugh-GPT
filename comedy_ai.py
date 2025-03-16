import os
import openai
import streamlit as st
import librosa
import numpy as np
from dotenv import load_dotenv
import whisper

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# Load whisper model on demand
@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        print(f"Failed to load Whisper model: {str(e)}")
        return None

def joke_feedback(joke_text):
    """Analyzes a joke using OpenAI and provides feedback on humor, structure, and clarity."""
    try:
        prompt = f"""
        You are a supportive comedy coach. Analyze this joke for humor, structure, and clarity.
        Provide constructive feedback and specific suggestions for improvement.
        Joke:
        "{joke_text}"
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a supportive comedy coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing joke: {str(e)}"

def transcribe_audio(audio_path):
    """Transcribe an audio file using OpenAI Whisper."""
    # Load the model on demand
    whisper_model = load_whisper_model()
    if not whisper_model:
        return "Failed to load transcription model"
    
    try:
        # Directly transcribe with whisper (it handles MP3/WAV automatically)
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Transcription error: {str(e)}"

def analyze_audio_metrics(audio_path):
    """Extracts audio metrics such as duration, speaking rate, number of pauses, and loudness."""
    try:
        # Load audio (librosa can handle MP3 and WAV)
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate speaking rate
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        estimated_words = len(onsets) / 3  # Approximation
        words_per_minute = (estimated_words / duration) * 60
        
        # Calculate pauses
        pauses = librosa.effects.split(y, top_db=25)
        num_pauses = len(pauses) - 1 if len(pauses) > 0 else 0
        
        # Calculate loudness
        rms = librosa.feature.rms(y=y).mean()
        normalized_loudness = float(np.clip(rms * 100, 0, 100))
        
        return {
            "duration_seconds": duration,
            "words_per_minute": words_per_minute,
            "num_pauses": num_pauses,
            "normalized_loudness": normalized_loudness
        }
    except Exception as e:
        print(f"Audio analysis error: {str(e)}")
        return None