import streamlit as st
from comedy_ai import joke_feedback, transcribe_audio, analyze_audio_metrics
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np

st.set_page_config(page_title="AI Comedy Coach", page_icon="🎤")

st.title("🎤 AI Comedy Coach")
st.write("Upload a joke in text or audio format and get feedback from our AI comedy coach!")

option = st.radio("Choose your input type:", ["Text", "Audio"])

if option == "Text":
    joke_text = st.text_area("Enter your joke:", height=150)
    if st.button("Analyze My Joke"):
        if joke_text.strip():
            with st.spinner("Analyzing your joke..."):
                feedback = joke_feedback(joke_text)
                st.markdown("## 📝 Feedback:")
                st.write(feedback)
        else:
            st.warning("Please enter a joke to analyze.")
            
elif option == "Audio":
    audio_file = st.file_uploader("Upload your audio file (MP3 or WAV)", type=["mp3", "wav"])
    
    if audio_file is not None:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        file_extension = audio_file.name.split('.')[-1].lower()
        temp_audio_path = os.path.join(temp_dir, f"temp_audio.{file_extension}")
        
        # Save uploaded file
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Process the audio
        with st.spinner("Transcribing your audio..."):
            transcript = transcribe_audio(temp_audio_path)
            st.markdown("### 🗣 Transcription:")
            st.write(transcript)
        
        with st.spinner("Analyzing your joke delivery..."):
            # Get joke feedback
            feedback = joke_feedback(transcript)
            st.markdown("## 📝 Joke Feedback:")
            st.write(feedback)
            
            # Analyze audio metrics
            audio_metrics = analyze_audio_metrics(temp_audio_path)
            
            if audio_metrics:
                st.markdown("## 🎙 Delivery Analysis:")
                st.write(f"- Duration: {audio_metrics['duration_seconds']:.2f} seconds")
                st.write(f"- Speaking Rate: {audio_metrics['words_per_minute']:.1f} words per minute")
                st.write(f"- Number of Pauses: {audio_metrics['num_pauses']}")
                st.write(f"- Voice Projection (0-100): {audio_metrics['normalized_loudness']:.1f}")
                
                # Create visualization with normalized values
                st.markdown("### 🎙️ Delivery Metrics:")
                
                # Normalize values for better visualization
                speaking_rate_norm = min(100, audio_metrics["words_per_minute"] / 2)  # Normalize against 200 wpm
                pauses_norm = min(100, audio_metrics["num_pauses"] * 10)  # Scale pauses
                loudness = audio_metrics["normalized_loudness"]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics = ["Speaking Rate", "Pauses", "Voice Projection"]
                values = [speaking_rate_norm, pauses_norm, loudness]
                colors = ['#4287f5', '#f5a742', '#42f584']
                
                bars = ax.barh(metrics, values, color=colors)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Score (normalized to 0-100)")
                ax.set_title("Comedy Delivery Metrics")
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                            ha='left', va='center')
                
                st.pyplot(fig)
                
                # Provide specific delivery feedback
                st.markdown("### 💬 Delivery Feedback:")
                if audio_metrics["words_per_minute"] < 120:
                    st.write("- Your speaking rate is relatively slow. Consider picking up the pace to keep the audience engaged.")
                elif audio_metrics["words_per_minute"] > 180:
                    st.write("- You're speaking quite fast. Try slowing down a bit to give your audience time to process the joke.")
                
                if audio_metrics["num_pauses"] < 3 and audio_metrics["duration_seconds"] > 15:
                    st.write("- You could benefit from adding more strategic pauses to build tension and emphasize punchlines.")
                elif audio_metrics["num_pauses"] > 10 and audio_metrics["duration_seconds"] < 60:
                    st.write("- You have frequent pauses. Consider making your delivery more fluid while keeping pauses for emphasis.")
                
                if audio_metrics["normalized_loudness"] < 40:
                    st.write("- Your voice projection could be stronger. Try speaking with more confidence and volume.")
                elif audio_metrics["normalized_loudness"] > 80:
                    st.write("- Your voice projection is very strong. Ensure you're not overwhelming, but great energy!")
        
        # Clean up temporary file
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except:
            pass  # Silent cleanup failure is acceptable