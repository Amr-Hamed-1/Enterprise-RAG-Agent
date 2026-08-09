"""Quick diagnosis: run all golden questions and report retrieval + grounding hits.

Usage:
    python scripts/diagnose.py [k] [top_n]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import TEST_QUESTIONS
from app.generator import generate_rag_response
from app.retriever import rewrite_query
from app.service import RAGService

service = RAGService()
service.initialize()


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    questions = json.loads(TEST_QUESTIONS.read_text(encoding="utf-8"))

    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"\n{'='*70}\n[{i}/{len(questions)}] {q}")
        print("=" * 70)

        rewritten = rewrite_query(q)
        print(f"  Rewritten: {rewritten}")

        docs = service.hybrid_search(rewritten, k=k, top_n=top_n)
        pages = []
        for doc in docs:
            page = doc.metadata.get("page", "N/A")
            pages.append(page)
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"    - page {page}: {preview}...")

        answer = generate_rag_response(q, docs)
        expected_num = item["expected_number"]
        expected_page = item["expected_page"]
        has_number = expected_num in answer
        has_page = expected_page in pages

        status = "PASS" if (has_number and has_page) else ("PARTIAL" if has_number else "FAIL")
        print(f"  -> expected number '{expected_num}' in answer: {has_number}")
        print(f"  -> expected page {expected_page} retrieved: {has_page} (got {pages})")
        print(f"  -> STATUS: {status}")

        results.append({
            "question": q,
            "category": item["category"],
            "expected_number": expected_num,
            "expected_page": expected_page,
            "retrieved_pages": [str(p) for p in pages],
            "has_number": has_number,
            "has_page": has_page,
            "status": status,
        })

    passed = sum(1 for r in results if r["status"] == "PASS")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    print(f"\n{'='*70}\nDIAGNOSIS SUMMARY: {passed}/{len(results)} full PASS, {partial} partial\n{'='*70}")

    out = TEST_QUESTIONS.parents[1] / "diagnosis_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
