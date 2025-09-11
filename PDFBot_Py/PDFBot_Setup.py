import logging
import os
import shutil
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from llama_index.llms.ollama import Ollama

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Centralized configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL", "llama3.1")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./embeddings")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def PDFBot_Setup(
    col_name: str,
    RESET_DB: bool = False,
):
    """
    Set up embedding and generation models along with a persistent Chroma collection.
    """
    logging.info(f"Using Embedding Model: {EMBEDDING_MODEL_NAME}")
    logging.info(f"Using Generation Model: {GENERATION_MODEL_NAME}")

    # Embedding + LLM
    embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    llm = Ollama(
        model=GENERATION_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=30.0,
        temperature=0,
    )

    # For Chroma compatibility
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=EMBEDDING_MODEL_NAME,
    )

    embeddings_path = Path(EMBEDDINGS_PATH)
    if RESET_DB and embeddings_path.exists():
        logging.info(f"Resetting database at {embeddings_path}")
        shutil.rmtree(embeddings_path)

    embeddings_path.mkdir(parents=True, exist_ok=True)

    # Persistent Chroma client
    chroma_client = chromadb.PersistentClient(
        path=str(embeddings_path),
        settings=Settings(
            anonymized_telemetry=False, allow_reset=True
        ),  # Disable telemetry
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # Create/get collection
    col = chroma_client.get_or_create_collection(
        name=col_name,
        embedding_function=ollama_ef,
    )

    logging.info(
        f"ChromaDB collection '{col_name}' is ready with {col.count()} documents."
    )

    return embed_model, llm, ollama_ef, col, chroma_client
