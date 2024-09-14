# Modification of code to work with Ollama
# Import Required Libraries

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.llms import Cohere
from langchain_cohere import CohereRerank
from langchain_chroma import Chroma 
import gradio as gr
import ollama
import os

from dotenv import load_dotenv
load_dotenv()

from PDFBot_Setup import PDFBot_Setup
from PDFBot_Load import PDFBot_Load, PDFBot_Store

compression_retriever = None
GENERATION_MODEL = None

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def chat(query, history):
    global compression_retriever, GENERATION_MODEL
    compressed_docs = compression_retriever.invoke(query)

    print(compressed_docs)

    SYSTEM_PROMPT = """
    You are a PDF expert assistant with a focus on accurate and reliable information retrieval from the documents provided to you. 
    You must only answer questions based on the content of these documents. 
    If you do not find the answer in the documents, respond with "I don't know." 
    Avoid providing speculative or unrelated information, and do not pull in knowledge from external sources beyond what is contained in the given documents. 
    Always prioritize correctness and clarity in your responses.
    """

    prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {SYSTEM_PROMPT}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Query: {query}
            Answer: Answer using {compressed_docs}<|eot_id|> 
            <|start_header_id|>assistant<|end_header_id|>
            """

    response = ollama.generate(system=SYSTEM_PROMPT, prompt=prompt, model=GENERATION_MODEL)['response']   

    return response

def main(path=None):
    global compression_retriever, GENERATION_MODEL
    name = "Resume"
    col_name = "LlamaParse"
    EMBEDDING_MODEL = "nomic-embed-text"
    GENERATION_MODEL = "llama3.1"

    # Set Up
    print("Starting")
    embed_model, llm, ollama_ef, col, chroma_client = PDFBot_Setup(col_name, EMBEDDING_MODEL = "nomic-embed-text", GENERATION_MODEL = "llama3.1")
    
    # Parsing PDF 
    print("Parsing PDF")
    base_nodes, objects = PDFBot_Load(name, llm, path=path)

    # Updating ChromaDB
    print("Updating ChromaDB")
    col_updated = PDFBot_Store(col=col, base_nodes=base_nodes, objects=objects, ollama_ef=ollama_ef)

    # Pass collection to langchain_chroma, instantiate retriever and cohere reranker
    print("Retrieval")
    db = Chroma(client=chroma_client, collection_name=col_name, embedding_function=embed_model)
    retriever = db.as_retriever()

    compressor = CohereRerank(top_n=10, model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )


with gr.Blocks() as demo:
    
    # First Block: File upload block
    with gr.Column(elem_id="upload_section") as upload_block:
        gr.Markdown("<h1 style='text-align: center;'>PDFBot</h1>")
        upload_btn = gr.UploadButton(file_count="single", label="Upload PDF", file_types=["file"])
    
    # Second Block: Chat interface, initially hidden
    with gr.Column(visible=False, elem_id="chat_section") as chat_block:
        chat_interface = gr.ChatInterface(fn=chat, title="PDFBot")
    
    # Function to switch to the second block (chat block)
    def switch_to_chat(file):
        if file:
            main(file)
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(), gr.update()
    
    # Set the upload button to trigger the switch
    upload_btn.upload(switch_to_chat, inputs=[upload_btn], outputs=[upload_block, chat_block])


demo.css = """
#upload_section {
    padding-top: 25%;
    justify-content: center;
    align-items: center;
}

#chat_section {
    height: 90vh;
}

.gr-chat-interface {
    height: calc(90vh - 50px); /* Full screen minus space for the title */
}
"""

# Launch the app
demo.launch()
