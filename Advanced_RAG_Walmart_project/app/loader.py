"""Phase 1-3: Load, chunk, and index the PDF corpus into Qdrant."""

import os
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.config import COLLECTION_NAME, PDF_PATH


def load_pdf(pdf_path: str) -> List[Document]:
    """Phase 1: Parse the PDF page-by-page with page numbers in metadata."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[Phase 1] Parsed {len(docs)} pages from {os.path.basename(pdf_path)}.")
    return docs


def chunk_documents(
    docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 150
) -> List[Document]:
    """Phase 2: Split documents into overlapping chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunked_docs = text_splitter.split_documents(docs)
    print(f"[Phase 2] Split {len(docs)} pages into {len(chunked_docs)} chunks.")
    return chunked_docs


def create_qdrant_vector_store(
    chunked_docs: List[Document], collection_name: str = COLLECTION_NAME
) -> QdrantVectorStore:
    """Phase 3: Embed and upload chunks to the Qdrant Cloud collection."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True},
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True,
        timeout=120,
    )
    print(f"[Phase 3] Uploaded {len(chunked_docs)} chunks to Qdrant '{collection_name}'.")
    return vector_store


if __name__ == "__main__":
    create_qdrant_vector_store(chunk_documents(load_pdf(str(PDF_PATH))))
