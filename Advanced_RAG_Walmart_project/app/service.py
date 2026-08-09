"""RAGService: orchestrates the full retrieval-augmented generation pipeline."""

import time
from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import COLLECTION_NAME, PDF_PATH
from app.generator import generate_rag_response
from app.loader import chunk_documents, load_pdf
from app.retriever import get_qdrant_vector_store, rewrite_query

FlashrankRerank.model_rebuild()


class RAGService:
    """Enterprise RAG pipeline exposed as a reusable service.

    Heavy components (Qdrant connection, PDF parsing, BM25 index, reranker)
    are initialized once at startup and reused across requests so the API
    stays fast and stateless.
    """

    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.collection_name = collection_name
        self.vector_store = None
        self.bm25_retriever = None
        self.reranker = None
        self.num_chunks = 0
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        if self._ready:
            return

        print("[RAGService] Connecting to Qdrant vector store...")
        self.vector_store = get_qdrant_vector_store(self.collection_name)

        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"PDF corpus not found at '{PDF_PATH}'. Make sure the data/ folder contains the report."
            )

        print("[RAGService] Loading and chunking the PDF corpus...")
        chunks = chunk_documents(load_pdf(str(PDF_PATH)))
        self.num_chunks = len(chunks)

        print("[RAGService] Building BM25 keyword index...")
        self.bm25_retriever = BM25Retriever.from_documents(documents=chunks)

        self.reranker = FlashrankRerank(top_n=5)

        self._ready = True
        print(f"[RAGService] Ready. {self.num_chunks} chunks indexed for hybrid search.")

    def hybrid_search(
        self, query: str, k: int = 10, top_n: int = 5, weights: tuple = (0.5, 0.5)
    ) -> List[Document]:
        """Dense (Qdrant) + sparse (BM25) ensemble search with FlashRank reranking."""
        if not self._ready:
            self.initialize()

        self.bm25_retriever.k = k
        qdrant_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        ensemble = EnsembleRetriever(
            retrievers=[qdrant_retriever, self.bm25_retriever],
            weights=list(weights),
        )

        retrieved_docs = ensemble.invoke(query)

        self.reranker.top_n = top_n
        reranked_docs = self.reranker.compress_documents(
            documents=retrieved_docs, query=query
        )

        print(
            f"[RAGService] Hybrid search: retrieved {len(retrieved_docs)} -> reranked top {len(reranked_docs)}"
        )
        return reranked_docs

    def answer(
        self, query: str, k: int = 10, top_n: int = 5
    ) -> dict:
        """Run the full pipeline and return a structured, serializable result."""
        if not self._ready:
            self.initialize()

        t_start = time.perf_counter()

        t_rewrite = time.perf_counter()
        rewritten_query = rewrite_query(query)
        rewrite_ms = (time.perf_counter() - t_rewrite) * 1000

        t_retrieval = time.perf_counter()
        top_chunks = self.hybrid_search(rewritten_query, k=k, top_n=top_n)
        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

        t_generate = time.perf_counter()
        answer = generate_rag_response(query=query, context_docs=top_chunks)
        generation_ms = (time.perf_counter() - t_generate) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        sources = [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source", self.collection_name),
            }
            for doc in top_chunks
        ]

        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources,
            "metrics": {
                "num_chunks_indexed": self.num_chunks,
                "query_rewrite_ms": round(rewrite_ms, 1),
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        }
