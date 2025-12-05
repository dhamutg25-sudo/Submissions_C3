from transformers import pipeline
import gradio as gr
import json
import tempfile

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

my_theme = gr.themes.Soft()

def read_summary(content):
    # Summarize the text
    summary = summarizer(content, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def download_output(content):
    # Save content to a temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.write(content.encode("utf-8"))
    tmp.close()
    return tmp.name  # return file path → Gradio downloads it


with gr.Blocks() as demo:
    input = gr.Textbox(label="Summary Text", lines=5)
    output = gr.Textbox(lines=5, label="Output")

    btn = gr.Button("Show Output")
    btn.click(read_summary, inputs=input, outputs=output)

    download_btn = gr.Button("Download Output")
    download_btn.click(download_output, inputs=output, outputs=gr.File(label="Download File"))

demo.launch(share=True) # share=True for public access URL