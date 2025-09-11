# simple_fast_pdfbot.py
import logging
import os
import pickle
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import fitz  # PyMuPDF
import gradio as gr
import ollama
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Configuration
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

# Application state
app_state = {"retriever": None}


def parse_pdf_local(pdf_path):
    """Parse PDF using PyMuPDF - fast and local."""
    documents = []

    # Check cache first
    cache_path = Path("./data") / f"cached_{Path(pdf_path).stem}.pkl"
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                logging.info("Loading cached PDF data")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Cache loading failed: {e}")

    # Parse PDF
    logging.info("Parsing PDF with PyMuPDF...")
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            documents.append(
                {"text": text, "metadata": {"source": pdf_path, "page": page_num}}
            )
    doc.close()

    # Cache results
    with open(cache_path, "wb") as f:
        pickle.dump(documents, f)

    logging.info(f"Parsed {len(documents)} pages")
    return documents


def setup_retriever(pdf_path):
    """Setup the retrieval system without reranking for maximum speed."""
    # Parse PDF
    documents = parse_pdf_local(pdf_path)

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Slightly larger for better context
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )

    texts = []
    metadatas = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            if chunk.strip():
                texts.append(chunk)
                metadatas.append(doc["metadata"])

    if not texts:
        raise ValueError("No text extracted from PDF")

    logging.info(f"Created {len(texts)} text chunks")

    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path="./embeddings")

    # Clear existing collection
    try:
        chroma_client.delete_collection("documents")
        logging.info("Cleared existing collection")
    except:
        pass

    # Create embedding function
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )

    collection = chroma_client.create_collection(
        name="documents", embedding_function=ollama_ef
    )

    # Add documents in batches for speed
    batch_size = 20
    logging.info("Adding documents to ChromaDB...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = [f"doc_{i + j}" for j in range(len(batch_texts))]

        collection.add(documents=batch_texts, metadatas=batch_metas, ids=batch_ids)

    logging.info("Documents added successfully")

    # Create retriever - using fewer documents for speed
    embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    db = Chroma(
        client=chroma_client,
        collection_name="documents",
        embedding_function=embed_model,
    )

    # Retrieve only top 6 most relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 10})

    return retriever


def format_docs(docs):
    """Format documents for context."""
    return "\n\n---\n\n".join(
        f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content}" for doc in docs
    )


def chat(message, history):
    """Fast chat function without reranking."""
    if not app_state.get("retriever"):
        yield "Please upload a PDF first."
        return

    try:
        # Get relevant documents (fast retrieval)
        docs = app_state["retriever"].get_relevant_documents(message)

        if not docs:
            yield "No relevant information found in the document."
            return

        # Format context (limit to prevent token overflow)
        context = format_docs(docs)
        if len(context) > 3000:  # Truncate if too long
            context = context[:3000] + "..."

        # System prompt
        system_prompt = """You are a helpful assistant. Answer the user's question based only on the provided document context. If you cannot find the answer in the context, say "I don't know based on the provided document." Be concise and accurate."""

        # Generate response
        prompt = f"Document Context:\n{context}\n\nUser Question: {message}"

        response = ""
        for chunk in ollama_client.chat(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            options={
                "num_predict": 400,  # Limit response length
                "temperature": 0.1,  # More focused responses
            },
        ):
            if chunk["message"]["content"]:
                token = chunk["message"]["content"]
                response += token
                yield response

    except Exception as e:
        logging.error(f"Chat error: {e}")
        yield f"Error processing your question: {str(e)}"


def process_pdf(file_path, progress=gr.Progress()):
    """Process uploaded PDF with progress tracking."""
    if not file_path:
        return gr.update(), gr.update()

    try:
        progress(0.1, desc="Starting PDF processing...")

        progress(0.3, desc="Parsing PDF content...")

        progress(0.6, desc="Setting up search index...")
        retriever = setup_retriever(file_path)
        app_state["retriever"] = retriever

        progress(0.9, desc="Finalizing setup...")

        progress(1.0, desc="Ready to chat!")

        return gr.update(visible=False), gr.update(visible=True)

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        gr.Error(f"Failed to process PDF: {str(e)}")
        return gr.update(), gr.update()


# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fast PDFBot") as demo:
    gr.Markdown("# 🚀 Fast Local PDFBot")
    gr.Markdown("Upload a PDF and chat with it instantly - no external APIs required!")

    with gr.Column(elem_id="upload_section") as upload_block:
        file_output = gr.File(
            label="📄 Upload PDF Document", file_types=[".pdf"], height=100
        )
        upload_button = gr.Button("🔄 Process PDF", variant="primary", size="lg")

        gr.Markdown("**Features:**")
        gr.Markdown("• Local processing (no data leaves your machine)")
        gr.Markdown("• Fast retrieval (3-8 second responses)")
        gr.Markdown("• Automatic caching for repeated use")

    with gr.Column(visible=False, elem_id="chat_section") as chat_block:
        gr.Markdown("### 💬 Chat with your PDF")
        chat_interface = gr.ChatInterface(
            fn=chat,
            examples=[
                "What is this document about?",
                "Summarize the main points",
                "What are the key findings?",
                "Are there any important dates mentioned?",
            ],
            title="Ask questions about your document",
            type="messages",
        )

    upload_button.click(
        fn=process_pdf, inputs=[file_output], outputs=[upload_block, chat_block]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        show_error=True,
    )
