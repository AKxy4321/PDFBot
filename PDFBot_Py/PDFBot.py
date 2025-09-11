# %%
import os

import gradio as gr
import ollama
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from PDFBot_Load import PDFBot_Load, PDFBot_Store
from PDFBot_Setup import PDFBot_Setup

load_dotenv()

# Use environment variables for model names
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.1")
COHERE_API_KEY = os.getenv("CO_API_KEY")

ollama_client = ollama.Client(
    host=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
)

# Application state for storing retriever and models
app_state = {"compression_retriever": None}


def format_docs(docs):
    """Helper function to format documents for the prompt context."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs
    )


def chat(query, history):
    """Chat function to handle user queries."""
    if not app_state.get("compression_retriever"):
        yield "Retriever not initialized. Please upload a PDF first."
        return

    retriever = app_state["compression_retriever"]
    compressed_docs = retriever.invoke(query)

    if not compressed_docs:
        yield "I could not find any relevant information in the document to answer your question."
        return

    context_str = format_docs(compressed_docs)

    SYSTEM_PROMPT = """You are an expert Q&A assistant. Your task is to answer the user's query based *only* on the provided context.
- If the context contains the answer, provide it clearly and concisely.
- If the context does not contain the answer, you MUST respond with "I don't know.".
- Do not use any external knowledge or make assumptions.
- Cite the source of the information if available."""

    prompt = f"Context:\n---\n{context_str}\n---\nUser Query: {query}"

    response = ""
    try:
        for chunk in ollama_client.chat(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        ):
            token = chunk["message"]["content"]
            response += token
            yield response
    except Exception as e:
        yield f"An error occurred with the language model: {e}"


def process_pdf(file_path, progress=gr.Progress(track_tqdm=True)):
    """Main processing pipeline for a given PDF file."""
    if not COHERE_API_KEY:
        raise gr.Error("COHERE_API_KEY environment variable not set.")

    col_name = "LlamaParse"

    progress(0, desc="Step 1/4: Initializing models and database...")
    embed_model, llm, _, col, chroma_client = PDFBot_Setup(col_name=col_name)

    progress(0.25, desc="Step 2/4: Parsing PDF with LlamaParse...")
    base_nodes, objects = PDFBot_Load(name="resume", llm=llm, path=file_path)

    progress(0.5, desc="Step 3/4: Storing and embedding document chunks...")
    # The PDFBot_Store function is now a generator that yields progress
    for _ in PDFBot_Store(
        col=col,
        base_nodes=base_nodes,
        objects=objects,
        embed_model=embed_model,
        stream=True,
    ):
        # This loop will now update the progress bar implicitly via track_tqdm
        pass

    progress(0.75, desc="Step 4/4: Setting up retrieval pipeline...")
    db = Chroma(
        client=chroma_client,
        collection_name=col_name,
        embedding_function=embed_model,
    )
    retriever = db.as_retriever(search_kwargs={"k": 20})

    compressor = CohereRerank(top_n=10, model="rerank-english-v3.0")
    app_state["compression_retriever"] = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    progress(1, desc="Processing complete. Ready to chat!")
    return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>PDFBot</h1>")

    with gr.Column(elem_id="upload_section") as upload_block:
        file_output = gr.File(label="Upload your PDF")
        upload_button = gr.Button("Process PDF", variant="primary")

    with gr.Column(visible=False, elem_id="chat_section") as chat_block:
        chat_interface = gr.ChatInterface(
            fn=chat,
            title="Chat with your PDF",
            description="Ask questions about the document you uploaded.",
        )

    upload_button.click(
        fn=process_pdf,
        inputs=[file_output],
        outputs=[upload_block, chat_block],
    )

if __name__ == "__main__":
    demo.launch()
