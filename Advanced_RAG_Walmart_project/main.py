import sys

from Retriver_Reranker import get_qdrant_vector_store , hybrid_search, rewrite_query 
from Generator import generate_rag_response



def run_enterprise_rag(user_query: str):
    """
    Executes the End-to-End Enterprise RAG Pipeline:
    1. Query Transformation (Rewriting via Llama 3.1)
    2. Hybrid Search (Dense Qdrant Vector + Sparse BM25)
    3. Cross-Encoder Re-ranking (Local FlashRank)
    4. Strict Financial Generation (Groq Llama 3.1)
    """
    print("\n" + "="*65)
    print(f"🚀 STARTING ENTERPRISE RAG PIPELINE")
    print(f"❓ User Query: '{user_query}'")
    print("="*65 + "\n")

    try:
        # 1. Connect to Qdrant Cloud Vector Store
        print("🔌 Connecting to Qdrant Cloud Collection...")
        vector_store = get_qdrant_vector_store("walmart_annual_report_2025")

        # 2. Phase 4A: Query Rewrite
        optimized_query = rewrite_query(user_query)

        # 3. Phase 4B & 4C: Hybrid Search + FlashRank Re-ranker
        top_chunks = hybrid_search(
            query=optimized_query,
            vector_store=vector_store,
            docs=None,
            k=10,
            top_n=3
        )

        # 4. Phase 5: Generation
        final_answer = generate_rag_response(
            query=user_query,
            context_docs=top_chunks
        )

        print("="*65)
        print("🎯 FINAL RAG ANSWER:")
        print("="*65)
        print(final_answer)
        print("\n" + "="*65)
        print("✅ Pipeline Executed Successfully!")
        print("="*65 + "\n")

    except Exception as e:
        print(f"\n❌ Error encountered during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # query which we are going to test on the pipeline
    sample_query = "What were Walmart's total revenues, operating income, and global e-commerce growth in fiscal year 2025?"
    run_enterprise_rag(sample_query)
    