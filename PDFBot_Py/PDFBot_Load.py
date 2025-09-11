import logging
import pickle
from pathlib import Path
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_parse import LlamaParse

logging.basicConfig(level=logging.INFO)


def PDFBot_Load(name, llm, path=None, force_reparse=False):
    """
    Load and parse a PDF into structured nodes.
    Uses pickle cache if available unless force_reparse=True.
    """
    parsing_instructions = """This document is a user-uploaded PDF. Your task is to meticulously extract all text, tables, and structural elements. Preserve the original formatting and layout as Markdown. Pay close attention to headings, lists, and data within tables."""

    data_dir = Path("./data")  # Use a local data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    if path is None:
        pdf_path = data_dir / f"{name}.pdf"
        pickle_path = data_dir / f"parsed_{name}_documents.pkl"
    else:
        pdf_path = Path(path)
        pickle_path = data_dir / f"parsed_{pdf_path.stem}_documents.pkl"

    documents = None
    if pickle_path.exists() and not force_reparse:
        try:
            with open(pickle_path, "rb") as f:
                documents = pickle.load(f)
            logging.info(f"Loaded cached documents from {pickle_path}")
        except (pickle.UnpicklingError, EOFError) as e:
            logging.warning(f"Cache file {pickle_path} is corrupted, reparsing: {e}")
            documents = None  # Ensure re-parsing happens

    if documents is None:
        parser = LlamaParse(
            result_type="markdown",
            system_prompt=parsing_instructions,
            num_workers=6,  # Use a safe number of workers
        )
        documents = parser.load_data(str(pdf_path))
        with open(pickle_path, "wb") as f:
            pickle.dump(documents, f)
        logging.info(f"Parsed and cached documents at {pickle_path}")

    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    return base_nodes, objects


def PDFBot_Store(col, base_nodes, objects, embed_model, batch_size=32, stream=False):
    """
    Store parsed nodes into Chroma with embeddings.
    Adds metadata (source, type) and batches embeddings for efficiency.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=50
    )

    all_nodes = base_nodes + objects
    if not all_nodes:
        logging.warning("No nodes to store.")
        if stream:
            yield 0
        return

    total_stored = 0
    for node in all_nodes:
        if not node.text.strip():
            continue

        split_texts = text_splitter.split_text(node.text)
        # No need for unique_texts, as we use hash-based IDs to prevent duplicates
        if not split_texts:
            continue

        ids = [str(hash(text)) for text in split_texts]
        metadatas = [
            {"source": node.ref_doc_id or "unknown", "type": type(node).__name__}
            for _ in split_texts
        ]

        # Batch embedding + storage
        for i in range(0, len(split_texts), batch_size):
            batch_texts = split_texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            embeddings = embed_model.embed_documents(batch_texts)

            col.upsert(
                documents=batch_texts,
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            total_stored += len(batch_texts)
            if stream:
                yield len(batch_texts)

    logging.info(f"Upserted {total_stored} chunks into {col.name}")
    if not stream:
        return col
