# pip install gtts
# TTS using google tts model
import gradio as gr
from gtts import gTTS
import tempfile

def tts_fn(text):
    tts = gTTS(text, lang="en")
    fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(fp.name)
    return fp.name

with gr.Blocks() as demo:
    gr.Markdown("## Text to Speech Demo")
    inp = gr.Textbox(label="Enter text")
    out = gr.Audio(label="Speech Output")
    btn = gr.Button("Convert")

    btn.click(tts_fn, inp, out)

demo.launch()