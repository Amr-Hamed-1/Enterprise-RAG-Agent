"""Retrieval tuning: sweep k / top_n over the golden questions.

Usage:
    python scripts/tune_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import TEST_QUESTIONS
from app.retriever import rewrite_query
from app.service import RAGService

service = RAGService()
service.initialize()

questions = json.loads(TEST_QUESTIONS.read_text(encoding="utf-8"))
rewards = {q["question"]: q["expected_page"] for q in questions}


def evaluate(k: int, top_n: int) -> float:
    hits = 0
    for item in questions:
        q = item["question"]
        rewritten = rewrite_query(q)
        docs = service.hybrid_search(rewritten, k=k, top_n=top_n)
        pages = [d.metadata.get("page") for d in docs]
        if item["expected_page"] in pages:
            hits += 1
    return hits / len(questions)


def main() -> None:
    print("Fixed rewrite (temperature=0) sweep:")
    for k in [10, 15, 20, 25]:
        for top_n in [3, 5]:
            score = evaluate(k, top_n)
            print(f"  k={k:<3} top_n={top_n} -> {score:.0%} page-hit rate")
    print("Done.")


if __name__ == "__main__":
    main()
