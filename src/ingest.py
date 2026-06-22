"""
ingest.py
----------
Loads documents from the data/ folder, splits them into chunks,
generates embeddings, and builds a FAISS vector index.

Run this once (and again any time you update data/*.txt) before
starting the chatbot:

    python src/ingest.py
"""

import os
import glob
from langchain_community.document_loaders import TextLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")


def load_documents():
    """Load all .txt files from the data directory."""
    documents = []
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {DATA_DIR}. Add your dataset files first."
        )

    for file_path in txt_files:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
        print(f"Loaded: {os.path.basename(file_path)}")

    return documents


def chunk_documents(documents):
    """Split documents into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks


def build_vectorstore(chunks):
    """Generate embeddings and build a FAISS vector index."""
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to: {VECTORSTORE_DIR}")

    return vectorstore


def main():
    print("=== ALFA: Building Knowledge Base ===\n")
    documents = load_documents()
    chunks = chunk_documents(documents)
    build_vectorstore(chunks)
    print("\n=== Done. You can now run the chatbot with: streamlit run src/app.py ===")


if __name__ == "__main__":
    main()
