import os
import sys
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank

FlashrankRerank.model_rebuild()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ==========================================
# 1. Connect to Existing Qdrant Cloud Collection
# ==========================================
def get_qdrant_vector_store(collection_name:str = "walmart_annual_report_2025") -> QdrantVectorStore:
    """
    Connects to the existing Qdrant Cloud collection and returns a QdrantVectorStore instance.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY not found in environment variables") 
    
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

    print(f"✅ [Phase 3 Complete] Connected to existing Qdrant vector store '{collection_name}'.")
    return vector_store



# ==========================================
# 2. Phase 4A: Query Transformation (Rewriting)
# ==========================================
def rewrite_query(query: str) -> str:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables") 
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    
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

   
    print(f"🔍 [Phase 4A Complete] Query Rewritten:")
    print(f"   Original: {query}")
    print(f"   Rewritten: {rewritten_query}\n")
    
    return rewritten_query




# Phase 4B: Hybrid Search (Keyword + Semantic) + FlashRank Reranker
def hybrid_search(query: str, vector_store: QdrantVectorStore, docs: List[Document] = None, k: int = 10, top_n: int = 5) -> List[Document]:
    """
    Performs hybrid search using both keyword (BM25) and semantic search,
    followed by FlashRank reranking to produce re-scored top results.
    
    Args:
        query: The search query
        vector_store: Qdrant vector store instance
        docs: Optional list of parsed Document chunks for BM25. If None, loaded automatically.
        k: Number of candidate results to retrieve per retriever before reranking
        top_n: Number of final reranked top results to return
    
    Returns:
        List of reranked Document objects
    """
    print(f"🔍 [Phase 4B] Performing Hybrid Search & FlashRank Reranking...")
    
    if docs is None:
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Walmart Annual Report 2025.pdf")
        if os.path.exists(pdf_path):
            from Walmart2025_RAG import load_and_parse_walmart_docs, chunking_docs
            raw_docs = load_and_parse_walmart_docs(pdf_path)
            docs = chunking_docs(raw_docs)
        else:
            raise ValueError(f"Document list 'docs' was not provided and PDF file was not found at '{pdf_path}'.")

    qdrant_retriver = vector_store.as_retriever(search_kwargs={"k": k})
    BM25_retriver = BM25Retriever.from_documents(documents=docs, k=k)

    ensemble_retriver = EnsembleRetriever(
        retrievers=[qdrant_retriver, BM25_retriver],
        weights=[0.5, 0.5],
    )

    retrieved_docs = ensemble_retriver.invoke(query)

    # Initialize FlashRank Reranker
    compressor = FlashrankRerank(top_n=top_n)
    reranked_docs = compressor.compress_documents(documents=retrieved_docs, query=query)

    print(f"✅ [Phase 4B Complete] Hybrid Search & Reranking Performed (Retrieved {len(retrieved_docs)} docs -> Reranked top {len(reranked_docs)} docs).")

    return reranked_docs




# ==========================================
# Testing Execution
# ==========================================    

def main():
    collection_name = "walmart_annual_report_2025"
    vector_store = get_qdrant_vector_store(collection_name)
    query = "What is walmart's financial performance in 2025?"
    
    # 1. Rewrite Query
    optimized_query = rewrite_query(query)
    
    # 2. Vector Search Retrieval Test
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(optimized_query)

    print(f"🔍 [Retrieval Results] Found {len(retrieved_docs)} relevant documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')} - Score: {doc.metadata.get('score', 'N/A')}")
        print(f"   Content Preview: {doc.page_content[:200]}...")

    # 3. Hybrid Search Test
    print("\n--- Testing Hybrid Search ---")
    hybrid_results = hybrid_search(optimized_query, vector_store,retrieved_docs, k=3)
    print(f"🔍 [Hybrid Search Results] Found {len(hybrid_results)} relevant documents:")
    for i, doc in enumerate(hybrid_results, 1):
        print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"   Content Preview: {doc.page_content[:200]}...")




if __name__ == "__main__":
    main()    