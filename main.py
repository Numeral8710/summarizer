import gradio as gr
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


LLM = OpenAI(temperature=0.5, openai_api_key=os.environ["OPENAI_API_KEY"])

# MapReduce (fast) prompts
MAP_PROMPT = """Write a summary of this chunk of text that includes the main points and any important details.
{text}
"""
COMBINE_PROMPT = """Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""

# Refine (slow) prompts
QUESTION_PROMPT = """Please provide a summary of the following text.
TEXT: {text}
SUMMARY:
"""


def summarize_pdf(pdf_file, use_refine, chunk_prompt, combine_prompt):
    loader = PyPDFLoader(pdf_file.name)
    docs = loader.load_and_split()
    if use_refine:
        chain = load_summarize_chain(
            LLM,
            chain_type="refine",
            question_prompt=PromptTemplate(template=chunk_prompt, input_variables=["text"]),
            refine_prompt=PromptTemplate(template=combine_prompt, input_variables=["text"]),
            return_intermediate_steps=True,
        )
    else:
        chain = load_summarize_chain(
            LLM,
            chain_type="map_reduce",
            map_prompt=PromptTemplate(template=chunk_prompt, input_variables=["text"]),
            combine_prompt=PromptTemplate(template=combine_prompt, input_variables=["text"]),
            return_intermediate_steps=True,
        )
    try:
        result = chain(docs)
        return [result["output_text"], "\n".join(result["intermediate_steps"])]
    except Exception as e:
        return [f"Error! #{e}", ""]


with gr.Blocks() as app:
    gr.Markdown("Pass in a PDF to summarize it")
    chunk_input = gr.TextArea(
        label="Prompt for each chunk", value=MAP_PROMPT, lines=3, max_lines=10
    )
    combine_input = gr.TextArea(
        label="Prompt to combine the output", value=COMBINE_PROMPT, lines=4, max_lines=10
    )
    speed = gr.components.Checkbox(label="Use refine approach (slower / more expensive!)")

    upload_button = gr.UploadButton(label="Upload a PDF")
    output_summary = gr.components.Textbox(label="Summary", lines=4, max_lines=50)
    summary_chunks = gr.components.Textbox(label="Summaries for each chunk", lines=4, max_lines=100)
    upload_button.upload(
        summarize_pdf,
        inputs=[upload_button, speed, chunk_input, combine_input],
        outputs=[output_summary, summary_chunks],
    )

app.launch(share=False, server_name="0.0.0.0")
