from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_parse import LlamaParse
import pickle
import os


def PDFBot_Load(name, llm):

    # Create Custom Parsing Instructions

    parsing_instructions = '''Answer questions using the information in this pdf and be precise. Avoid Hallucinations, and say you don't know if given data is not enough to answer the question'''

    path = os.path.join('..', 'data', f'{name}.pdf')
    pickle_path = os.path.join('..', 'data', f'parsed_{name}_documents.pkl')

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as file:
            documents= pickle.load(file)
            print("Loaded documents")
    else:
        documents = LlamaParse(result_type="markdown", parsing_instructions=parsing_instructions).load_data(path)
        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(documents, pickle_file)

    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()

    # Retrieve nodes (text) and objects (table)

    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    # Check output of extraction

    # print(base_nodes)
    # print()
    # print(objects)

    return base_nodes, objects


def PDFBot_Store(col, base_nodes, objects, ollama_ef):
    # Store embeddings and metadata into chroma

    for node in base_nodes + objects:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[
            "\n",
            "\n\n"
        ])
        split_texts = text_splitter.split_text(node.text)
        i = 0
        for text in split_texts:
            doc = ollama_ef(text)
            col.add(documents=text, ids=str(i), embeddings=doc)
            i += 1

    return col