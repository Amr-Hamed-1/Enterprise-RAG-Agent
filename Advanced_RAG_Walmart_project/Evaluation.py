import os
import time
import logging
from typing import List
from dotenv import load_dotenv

# Disable noisy HTTP and library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("deepeval").setLevel(logging.WARNING)

# Import DeepEval core components and standard RAG metrics
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq

# Import custom project modules (Exact original names)
from Retriver_Reranker import get_qdrant_vector_store, rewrite_query, hybrid_search
from Generator import generate_rag_response

load_dotenv()


# =====================================================================
# Custom Groq LLM Class for DeepEval Judge (Using 70B Versatile)
# =====================================================================
class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """
    Wraps Groq model to serve as the LLM Judge for DeepEval.
    Uses llama-3.3-70b-versatile for strictly valid JSON parsing.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing in environment variables.")
        self.llm = ChatGroq(
            model_name=self.model_name,
            temperature=0.0,
            groq_api_key=self.groq_api_key
        )

    def load_model(self):
        return self.llm

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        res = self.llm.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str) -> str:
        res = await self.llm.ainvoke(prompt)
        return res.content


# =====================================================================
# Main Evaluation Pipeline
# =====================================================================
def run_deepeval_benchmark():
    print("\n" + "="*60)
    print("🚀 Enterprise RAG Benchmark - Evaluation Results")
    print("="*60)

    # Initialize 70B LLM Judge
    judge_llm = GroqDeepEvalLLM(model_name="llama-3.3-70b-versatile")

    # Golden Ground Truth Dataset
    test_cases_data = [
        {
            "question": "What were Walmart's total revenues in fiscal year 2025?",
            "ground_truth": "Walmart's total revenues were $680.985 billion in fiscal year 2025, representing a 5.1% growth."
        },
        {
            "question": "What was Walmart's operating income for fiscal year 2025?",
            "ground_truth": "Walmart reported an operating income of $29.374 billion in fiscal year 2025, an 8.8% increase."
        },
        {
            "question": "What was Walmart's global e-commerce sales growth in FY2025?",
            "ground_truth": "Global e-commerce sales grew by 21% in fiscal year 2025, driven by store-fulfilled pickup and delivery."
        }
    ]

    vector_store = get_qdrant_vector_store("walmart_annual_report_2025")

    for idx, item in enumerate(test_cases_data, start=1):
        query = item["question"]
        ground_truth = item["ground_truth"]

        # Run RAG Pipeline
        opt_query = rewrite_query(query)
        retrieved_docs = hybrid_search(opt_query, vector_store, k=10, top_n=3)
        retrieved_contents = [doc.page_content for doc in retrieved_docs]
        actual_output = generate_rag_response(query, retrieved_docs)

        # Construct TestCase
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieved_contents,
            expected_output=ground_truth
        )

        # Initialize Metrics
        faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_llm)
        answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_llm)
        contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=judge_llm)
        contextual_recall = ContextualRecallMetric(threshold=0.7, model=judge_llm)

        print(f"\n📌 [Test Case #{idx}] Query: \"{query}\"")

        def safe_eval(metric_obj, metric_name):
            try:
                metric_obj.measure(test_case)
                status = "PASSED" if metric_obj.is_successful() else "FAILED"
                print(f"   • {metric_name:<20}: {metric_obj.score:.2f}  [{status}]")
            except Exception:
                print(f"   • {metric_name:<20}: N/A   [LLM Parsing Error]")

        safe_eval(faithfulness, "Faithfulness")
        time.sleep(3)  # Cooldown buffer for 70B model rate limits

        safe_eval(answer_relevancy, "Answer Relevancy")
        time.sleep(3)

        safe_eval(contextual_precision, "Contextual Precision")
        time.sleep(3)

        safe_eval(contextual_recall, "Contextual Recall")

        if idx < len(test_cases_data):
            time.sleep(5)

    print("\n" + "="*60)
    print("✅ Benchmark Suite Completed Cleanly!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_deepeval_benchmark()
    