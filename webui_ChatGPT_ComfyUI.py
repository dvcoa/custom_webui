#import torch
#from openai import OpenAI
import whisper
import numpy as np
import gradio as gr
#from gradio.data_classes import FileData
#import tempfile
#import soundfile as sf
from gtts import gTTS
from io import BytesIO
import base64
import time
from transformers import MarianMTModel, MarianTokenizer
import ComfyUIAPI
import json
import html2text

# Load tokenizer and model (using 'Helsinki-NLP/opus-mt-ko-en' as an example)
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
marianmt_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

# Function to process user input
def process_input(user_input):
    # Check if the input starts with /image
    if user_input.startswith("/image"):
        # Remove the command part and strip any extra spaces
        command_removed = user_input[len("/image"):].strip()
        
        # Split the remaining input based on the delimiter "!"
        parts = command_removed.split('!', 1)
        
        # First argument is always present
        first_arg = parts[0].strip()
        
        # Second argument is optional
        second_arg = parts[1].strip() if len(parts) > 1 else None
        
        # Return the result as a tuple
        return (first_arg, second_arg) if second_arg else (first_arg,"")
        #return [first_arg, second_arg] if second_arg else [first_arg]
    else:
        return "Invalid command"

# Function to process text through neural machine translation (NMT)
def NMT_process(input_text):
    # Given English text
    #input_text = "He can goed to the store."
    print("input_text :", input_text)

    # Tokenize the text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Convert model input to translation format
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate translated output
    translated = marianmt_model.generate(input_ids=input_ids, attention_mask=attention_mask)
    corrected_text = tokenizer.decode(translated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("Corrected Text:", corrected_text)
    return corrected_text

# Initialize OpenAI client
#server_address = "http://localhost:1234/v1"
#client = OpenAI(base_url=server_address, api_key="lm-studio")

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize ChatGPT driver
import ChatGPT
driver = ChatGPT.webdriver_init()

# Function to handle AI response using ChatGPT
def AIAnswer(transcript):
    try:
        answer = ChatGPT.QueryChatGpt(driver,transcript)
        
    except Exception as e:
        print("Error:", e)
        return None
    
    return answer

# Function to convert text to speech audio
def text_to_speech(text):
    tts = gTTS(text[:200])
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<br/><audio src="data:audio/mpeg;base64,{audio}" controls autoplay"></audio>'

    return audio_player

# Function to add message to history
def add_message(history, message):
    print(message)
    for x in message["files"]:
        history.append(((x,), None))
        print(history)
    if message["text"] is not None:
        history.append((message["text"], None))        
        #history.append((('C:\\Users\\baehw\\AppData\\Local\\Temp\\gradio\\0fd1870571e7d163df58f63aa254f20e0e84074c\\ComfyUI_00043_.png',), None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

# Function to handle audio input and transcribe to text
def send_audio_message(history, audio):
    transcribed_user_message = whisper_model.transcribe(audio)
    if transcribed_user_message["text"] is not None:
        history.append((transcribed_user_message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

# Function to add audio message
def add_audio_message(audio):
    transcribed_user_message = whisper_model.transcribe(audio)
    if transcribed_user_message["text"] is not None:
        result = NMT_process(transcribed_user_message["text"])
        return gr.MultimodalTextbox(value={'text': result, 'files': None}, interactive=True)

# Function to handle bot's response with audio interaction
def bot_audio(history):
    if history[-1][0] is not None:
        # /image user prompt, negative prompt 
        if history[-1][0].startswith("/image"):
            user_prompt, negative_prompt = process_input(history[-1][0])
            print(user_prompt + negative_prompt)
            f = open('./base_workflow.json')
            file_path = ComfyUIAPI.prompt_to_image(f.read(),user_prompt,negative_prompt)     
            f.close()       
            history[-1][1] = (file_path,)
            #history[-1][1] = ("https://gradio-builds.s3.amazonaws.com/diffusion_image/cute_dog.jpg",)
            yield history, ""
        else:
            response = AIAnswer(history[-1][0]) # process msg
            response = response + text_to_speech(html2text.html2text(response)) 
            history[-1][1] = response
            yield history

# Function to handle bot's response with text interaction
def bot_txt(history):
    if history[-1][0].startswith("/image"):        
        return history
    else:
        response = history[-1][1]
        history[-1][1] = ""        
        for character in response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history

# Define Gradio Blocks interface
with gr.Blocks() as demo:


    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        height=700,
        bubble_full_width=False,
        sanitize_html=False
        #render_markdown = False
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot_audio, [chatbot], [chatbot], api_name="bot_response")
    #bot_msg = bot_audio.then(bot_txt, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    audio_input = gr.Microphone(type="filepath", interactive="True", label="Speak to Transcribe")
    audio_msg = audio_input.stop_recording(add_audio_message, [audio_input], [chat_input])

    mic_button = gr.Button(value="Mic off")
    mic_button.click(record)

# Launch Gradio interface
demo.queue()
#demo.launch(share=True)

# Open web interface in default browser
webui_address = "http://localhost:7860"
import webbrowser
webbrowser.open(webui_address)

demo.launch()
