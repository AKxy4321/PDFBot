# Modification of code to work with Ollama
# Import Required Libraries

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.llms import Cohere
from langchain_cohere import CohereRerank
from langchain_chroma import Chroma 
from uuid import uuid4
import gradio as gr
import ollama
import os

from dotenv import load_dotenv
load_dotenv()

from PDFBot_Setup import PDFBot_Setup
from PDFBot_Load import PDFBot_Load, PDFBot_Store


def chat(query, history):
    global compression_retriever, GENERATION_MODEL, chat_history
    compressed_docs = compression_retriever.invoke(query)

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

    response = ollama.generate(prompt=prompt, model=GENERATION_MODEL)['response']   

    return response

name = "Resume"
col_name = "LlamaParse"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.1"

embed_model, llm, ollama_ef, col, chroma_client = PDFBot_Setup(col_name, EMBEDDING_MODEL = "nomic-embed-text", GENERATION_MODEL = "llama3.1")
base_nodes, objects = PDFBot_Load(name, llm)

col_updated = PDFBot_Store(col=col, base_nodes=base_nodes, objects=objects, ollama_ef=ollama_ef)

# Pass collection to langchain_chroma, instantiate retriever and cohere reranker

db = Chroma(client=chroma_client, collection_name=col_name, embedding_function=embed_model)
retriever = db.as_retriever()

compressor = CohereRerank(top_n=10, model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

demo = gr.ChatInterface(fn=chat, title="PDFBot")
demo.launch()