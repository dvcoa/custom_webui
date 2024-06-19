import torch
from openai import OpenAI
import whisper
import numpy as np
import gradio as gr
from gradio.data_classes import FileData
import tempfile
import soundfile as sf
from gtts import gTTS
from io import BytesIO
import base64
import time
from transformers import MarianMTModel, MarianTokenizer
import ComfyUIAPI
import json

# 모델 및 토크나이저 로드 (예시로 'Helsinki-NLP/opus-100-en' 모델 사용)

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
marianmt_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")



f = open('./base_workflow.json')

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

# Example usage
#user_input = input("Enter your command: ")





def NMT_process(input_text):
    # 주어진 영어 텍스트
    #input_text = "He can goed to the store."
    print("input_text :", input_text)

    # 텍스트를 토큰화
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # 모델 입력을 번역 형식에 맞게 변환
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # 번역된 결과 생성
    translated = marianmt_model.generate(input_ids=input_ids, attention_mask=attention_mask)
    corrected_text = tokenizer.decode(translated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("Corrected Text:", corrected_text)
    return corrected_text

# 번역된 텍스트 디코딩하여 출력


server_address = "http://localhost:1234/v1"
# Initialize OpenAI client
client = OpenAI(base_url=server_address, api_key="lm-studio")

# Load Whisper model
whisper_model = whisper.load_model("base")



def AIAnswer(transcript):
    try:
        # Call OpenAI chat completions API
        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct-gguf",
            messages=[
                {"role": "system", "content": """
                Your name is Jenny,
                You are here for English teaching, so correct any errors and suggest better expressions.
                Your personality is very kind and intellectual, with deep knowledge, especially in technology.
                Always answer in short and no emoticons.
                """},
                {"role": "user", "content": transcript}
            ],
            temperature=0.7,
        )

        if not completion.choices:
            raise Exception("No response from Chat Server")

        answer = completion.choices[0].message.content

    except Exception as e:
        print("Error:", e)
        return None
    return answer

def text_to_speech(text):
    tts = gTTS(text)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'

    return audio_player

def add_message(history, message):
    print(message)
    for x in message["files"]:
        history.append(((x,), None))
        print(history)
    if message["text"] is not None:
        history.append((message["text"], None))        
        #history.append((('C:\\Users\\baehw\\AppData\\Local\\Temp\\gradio\\0fd1870571e7d163df58f63aa254f20e0e84074c\\ComfyUI_00043_.png',), None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

# New function to handle audio input and transcribe it to text
def send_audio_message(history, audio):
    transcribed_user_message = whisper_model.transcribe(audio)
    if transcribed_user_message["text"] is not None:
        history.append((transcribed_user_message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)
    
def add_audio_message(audio):
    transcribed_user_message = whisper_model.transcribe(audio)
    if transcribed_user_message["text"] is not None:
        result = NMT_process(transcribed_user_message["text"])
        return gr.MultimodalTextbox(value={'text': result, 'files': None}, interactive=True)

def bot_audio(history):
    if history[-1][0] is not None:
        # /image user prompt, negative prompt 
        if history[-1][0].startswith("/image"):
            user_prompt, negative_prompt = process_input(history[-1][0])
            print(user_prompt + negative_prompt)
            file_path = ComfyUIAPI.prompt_to_image(f.read(),user_prompt,negative_prompt)            
            history[-1][1] = (file_path,)
            #history[-1][1] = ("https://gradio-builds.s3.amazonaws.com/diffusion_image/cute_dog.jpg",)
            return history, ""
        else:
            response = AIAnswer(history[-1][0]) # process msg
            history[-1][1] = response
            return history, text_to_speech(response)         
                    


def bot_txt(history):    
    response = history[-1][1]
    yield history
    """
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history
    """


# Define Gradio Blocks interface
with gr.Blocks() as demo:

    html = gr.HTML()
    # html.visible = False

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_audio = chat_msg.then(bot_audio, [chatbot], [chatbot, html], api_name="bot_voice")
    #bot_msg = bot_audio.then(bot_txt, chatbot, chatbot, api_name="bot_response")
    #bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    audio_input = gr.Microphone(type="filepath", interactive="True", label="Speak to Transcribe")
    audio_msg = audio_input.stop_recording(add_audio_message, [audio_input], [chat_input])

# Launch Gradio interface
demo.queue()
#demo.launch(share=True)

webui_address = "http://localhost:7860"
import webbrowser
webbrowser.open(webui_address)

demo.launch()
