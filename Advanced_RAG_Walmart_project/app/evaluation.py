"""DeepEval benchmark: 4 RAG metrics judged by llama-3.3-70b on Groq."""

import logging
import os
import time

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_groq import ChatGroq

from app.generator import generate_rag_response
from app.retriever import rewrite_query
from app.service import RAGService

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("deepeval").setLevel(logging.WARNING)

service = RAGService()
service.initialize()


# =====================================================================
# Groq LLM Judge for DeepEval (70B versatile for strict JSON parsing)
# =====================================================================
class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing in environment variables.")
        self.llm = ChatGroq(
            model_name=self.model_name,
            temperature=0.0,
            groq_api_key=self.groq_api_key,
        )

    def load_model(self):
        return self.llm

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.llm.ainvoke(prompt)).content


# =====================================================================
# Golden ground-truth dataset (validated against the corpus)
# =====================================================================
TEST_CASES = [
    {
        "question": "What were Walmart's total revenues in fiscal year 2025?",
        "ground_truth": "Walmart's total revenues were $680,985 million in fiscal year 2025.",
    },
    {
        "question": "What were Walmart's net sales in fiscal year 2025?",
        "ground_truth": "Walmart's net sales were $674,538 million in fiscal year 2025.",
    },
    {
        "question": "What was Walmart's operating income for fiscal year 2025?",
        "ground_truth": "Walmart reported an operating income of $29,348 million in fiscal year 2025.",
    },
    {
        "question": "What was Walmart International's net sales in fiscal year 2025?",
        "ground_truth": "Walmart International's net sales were $121,885 million in fiscal year 2025.",
    },
    {
        "question": "How much did Walmart International e-commerce contribute to net sales in fiscal year 2025?",
        "ground_truth": "Approximately $29.5 billion of Walmart International's net sales related to eCommerce for fiscal 2025.",
    },
    {
        "question": "How much were Sam's Club U.S. grocery net sales in fiscal year 2025?",
        "ground_truth": "Sam's Club U.S. grocery net sales were $59,976 million in fiscal year 2025.",
    },
]


def run_deepeval_benchmark():
    print("\n" + "=" * 60)
    print("Enterprise RAG Benchmark - Evaluation Results")
    print("=" * 60)

    judge_llm = GroqDeepEvalLLM(model_name="llama-3.3-70b-versatile")

    for idx, item in enumerate(TEST_CASES, start=1):
        query = item["question"]
        ground_truth = item["ground_truth"]

        opt_query = rewrite_query(query)
        retrieved_docs = service.hybrid_search(opt_query, k=10, top_n=5)
        retrieved_contents = [doc.page_content for doc in retrieved_docs]
        actual_output = generate_rag_response(query, retrieved_docs)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieved_contents,
            expected_output=ground_truth,
        )

        metrics = {
            "Faithfulness": FaithfulnessMetric(threshold=0.7, model=judge_llm),
            "Answer Relevancy": AnswerRelevancyMetric(threshold=0.7, model=judge_llm),
            "Contextual Precision": ContextualPrecisionMetric(threshold=0.7, model=judge_llm),
            "Contextual Recall": ContextualRecallMetric(threshold=0.7, model=judge_llm),
        }

        print(f"\n[Test Case #{idx}] Query: {query!r}")

        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                status = "PASSED" if metric.is_successful() else "FAILED"
                print(f"   {name:<22}: {metric.score:.2f}  [{status}]")
            except Exception:
                print(f"   {name:<22}: N/A   [LLM Parsing Error]")
            time.sleep(3)  # cooldown buffer for 70B model rate limits

        if idx < len(TEST_CASES):
            time.sleep(5)

    print("\n" + "=" * 60)
    print("Benchmark suite completed.")
    print("=" * 60)


if __name__ == "__main__":
    run_deepeval_benchmark()
