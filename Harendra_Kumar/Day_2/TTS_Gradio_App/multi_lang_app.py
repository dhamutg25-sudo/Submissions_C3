# pip install gtts
# TTS using google tts model
import gradio as gr
from gtts import gTTS
import tempfile

def tts_fn(text, lang):
    fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    gTTS(text=text, lang=lang).save(fp.name)
    return fp.name

with gr.Blocks() as demo:
    gr.Markdown("## gTTS Text-to-Speech")

    text = gr.Textbox(label="Enter Text")
    lang = gr.Dropdown(["en", "es", "fr", "de", "hi", "ar"], value="en", label="Language")
    audio = gr.Audio(label="Generated Speech")
    btn = gr.Button("Convert")

    btn.click(tts_fn, [text, lang], audio)

demo.launch()