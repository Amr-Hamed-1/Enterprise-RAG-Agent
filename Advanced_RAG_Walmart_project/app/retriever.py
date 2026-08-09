"""Query transformation (rewrite) + Qdrant connection."""

import os

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import COLLECTION_NAME

FlashrankRerank.model_rebuild()


def get_qdrant_vector_store(collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
    """Connect to the existing Qdrant Cloud collection."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True},
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    print(f"[Retriever] Connected to Qdrant vector store '{collection_name}'.")
    return vector_store


def rewrite_query(query: str) -> str:
    """Phase 4A: Rewrite the query for better BM25 + vector retrieval."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.0)

    prompt_template = """
    <system>
    You are an expert financial search query optimizer for RAG systems.

    Your goal:
    - Rewrite the query to improve BM25 keyword and vector search accuracy.
    - Add relevant financial terminology or synonyms (e.g., FY2025, annual report, net sales) if helpful.
    - Keep the output extremely concise (strictly under 20 words / 1 short sentence).

    CRITICAL RULES:
    - DO NOT add sub-questions, comparisons, or extra scope not present in the original query.
    - DO NOT answer the question or output markdown explanation.
    - Preserve the original intent 100%.
    </system>

    <user_query>
    {query}
    </user_query>

    <instructions>
    Output ONLY the concise, optimized search query text:
    </instructions>
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    rewritten_query = chain.invoke({"query": query}).strip().replace('"', '')
    return rewritten_query
