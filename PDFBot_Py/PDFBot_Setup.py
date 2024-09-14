from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb.utils.embedding_functions as embedding_functions
from langchain_ollama import OllamaEmbeddings
from llama_index.llms.ollama import Ollama 
import chromadb
import shutil
import os

def PDFBot_Setup(col_name, EMBEDDING_MODEL = "nomic-embed-text", GENERATION_MODEL = "llama3.1", reset=True):

    # Instantiate the embedding and generation models

    # embed_model and llm here needs to be used with llama-index functions

    embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    llm = Ollama(model=GENERATION_MODEL, request_timeout=5.0, temperature=0)

    # Ollama_ef to be used with chromadb

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://127.0.0.1:11434/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )

    embeddings_path = os.path.join('..', 'embeddings')
    shutil.rmtree(embeddings_path)
    os.makedirs(embeddings_path)

    # Setup a chroma client, make it persistent so that we can store the embeddings
    chroma_client = chromadb.PersistentClient(path=embeddings_path,     
                                settings=Settings(allow_reset=True),
                                tenant=DEFAULT_TENANT,
                                database=DEFAULT_DATABASE,
                                )

    chroma_client.reset()

    # Define and instantiate the collection for the chromadb
    col = chroma_client.get_or_create_collection(col_name, embedding_function=ollama_ef)

    return embed_model, llm, ollama_ef, col, chroma_client