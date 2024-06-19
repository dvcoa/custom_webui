import torch
from TTS.api import TTS
from openai import OpenAI
import pygame
import time
import whisper
import pyaudio
import wave
import keyboard
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True).to(device)

pygame.mixer.init()

model = whisper.load_model("base")

def play_audio(file_path):
    print('Press ESC to stop playback.')
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():        
        if keyboard.is_pressed("esc"):  
            pygame.mixer.music.stop()  
            break
        time.sleep(0.01)
    pygame.mixer.music.unload()  

def text_to_speech(transcript):    
    try:
        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct-gguf",
            messages=[
                {"role": "system", "content": """
                Your name is Jenny,
                Your personality is very kind and intellectual, with deep knowledge, especially in technology.
                You are here for English teaching, so correct any errors and suggest better expressions.
                Always answer in short and no emoticons.
                """},
                {"role": "user", "content": transcript}
            ],
            temperature=0.7,
        )

        if not completion.choices:
            raise Exception("No response from Chat Server")

        answer = completion.choices[0].message.content

        if answer is None:
            answer = {"text": "Sorry, Chat Server doesn't respond"}

        tts.tts_to_file(text=answer, file_path='output.wav')

        play_audio('output.wav')

        return transcript

    except Exception as e:
        print("Error:", e)
        return None

def is_silent(data, threshold=500):
    """Return 'True' if below the 'silent' threshold"""
    return np.max(np.frombuffer(data, dtype=np.int16)) < threshold

def record_audio(output_file, sample_rate=44100, chunk_size=1024):
    audio_format = pyaudio.paInt16  
    channels = 2  

    audio = pyaudio.PyAudio()

    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Press Enter to start recording.")

    frames = []

    keyboard.wait("enter")
    print("Recording started...")

    while True:
        data = stream.read(chunk_size)
        frames.append(data)
        if keyboard.is_pressed("esc"):  
            break
    print("Recording stopped.")    

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"File saved as {output_file}")

def main():
    while True:
        record_audio("input.wav")
        result = model.transcribe("input.wav")
        print("User said:", result["text"])        
        ans = text_to_speech(result["text"])
        if ans and "exit Jenny" in ans:
            break

if __name__ == "__main__":
    main()
