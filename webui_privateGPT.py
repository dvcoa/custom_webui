import torch
#from openai import OpenAI
from pgpt_python.client import PrivateGPTApi
import whisper
import numpy as np
import gradio as gr
import tempfile
import soundfile as sf
from gradio.data_classes import FileData
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.events import Events

# Check for available device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize OpenAI client
#client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
client = PrivateGPTApi(base_url="http://localhost:8001")

# Load Whisper model
whisper_model = whisper.load_model("base")

from gtts import gTTS
from io import BytesIO
import base64
import wavio

def AIAnswer(transcript):
    try:
        result = client.contextual_completions.prompt_completion(
            prompt=transcript,
            use_context=True,
            context_filter={"docs_ids": ["aa167e99-f364-45db-9f3e-787437953cbc"]},
            include_sources=True,
        ).choices[0]        

        answer = result.message.content

    except Exception as e:
        print("Error:", e)
        return None
    return answer

def text_to_speech(text):
    tts = gTTS(text)
    tts.save('hello_world.mp3')

    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'

    return audio_player

def audio_to_html(audio):
    audio_bytes = BytesIO()
    wavio.write(audio_bytes, audio[1].astype(np.float32), audio[0], sampwidth=4)
    audio_bytes.seek(0)

    audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<audio src="data:audio/mpeg;base64,{audio_base64}" controls autoplay></audio>'

    return audio_player


# Define Gradio Blocks interface
with gr.Blocks() as demo:

    html = gr.HTML()
    # html.visible = False
    chatbot = MultimodalChatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )
    msg = gr.Textbox()
    
    def respond(message, chat_history):        
        if message is not None:
            user_message = {"text": message,"files": []}
            answer = AIAnswer(message) # process msg
            bot_response = {"text": answer,"files": []}
        chat_history.append((user_message, bot_response))
        #time.sleep(2)
        return "", chat_history, text_to_speech(answer)

# New function to handle audio input and transcribe it to text
    def audio_response(audio, history):
        transcribed_user_message = whisper_model.transcribe(audio)        
        if transcribed_user_message is not None:
            user_message = {"text": transcribed_user_message["text"],"files": []}
            answer = AIAnswer(transcribed_user_message["text"]) # process msg
            bot_response = {"text": answer,"files": []}
            history.append((user_message, bot_response))
                  
        return history, text_to_speech(answer)
    # chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

    msg.submit(respond, [msg, chatbot], [msg, chatbot, html])    
    audio_input = gr.Audio(sources="microphone", type="filepath", label="Speak to Transcribe")    
    audio_msg = audio_input.stop_recording(audio_response, [audio_input, chatbot], [chatbot, html])
    #audio_msg.then(bot, chatbot, chatbot, api_name="bot_response")

# Launch Gradio interface
demo.queue()
#demo.launch(share=True)
demo.launch()
