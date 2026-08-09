"""Enterprise RAG Agent — command-line entry point.

Usage:
    python main.py "What were Walmart's total revenues in fiscal year 2025?"
    python main.py "How much were Sam's Club grocery net sales?" --k 10 --top-n 5
"""

import argparse

from app.service import RAGService

DEFAULT_QUERY = (
    "What were Walmart's total revenues, operating income, and net sales in fiscal year 2025?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Enterprise RAG pipeline.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="Question about the report")
    parser.add_argument("--k", type=int, default=10, help="Candidates per retriever")
    parser.add_argument("--top-n", type=int, default=5, help="Final reranked chunks for the LLM")
    args = parser.parse_args()

    service = RAGService()
    service.initialize()
    result = service.answer(args.query, k=args.k, top_n=args.top_n)

    print("\n" + "=" * 65)
    print(f"Question: {result['query']}")
    print(f"Rewritten: {result['rewritten_query']}")
    print("=" * 65)
    print(result["answer"])
    print("=" * 65)
    m = result["metrics"]
    print(f"Sources: {len(result['sources'])} chunks")
    print(
        f"Timing: {m['total_ms']:.0f} ms total "
        f"(rewrite {m['query_rewrite_ms']:.0f} · retrieval {m['retrieval_ms']:.0f} · generation {m['generation_ms']:.0f})"
    )


if __name__ == "__main__":
    main()
