"""Phase 5: Grounded answer generation with Groq."""

import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


def format_docs(docs: List[Document]) -> str:
    """Format retrieved chunks into a clean context string."""
    formatted_chunks = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Walmart Annual Report 2025")
        page = doc.metadata.get("page", "N/A")
        formatted_chunks.append(
            f"--- CHUNK {idx} (Source: {source} | Page: {page}) ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def generate_rag_response(query: str, context_docs: List[Document]) -> str:
    """Generate a strictly grounded answer using only the provided context."""
    formatted_context = format_docs(context_docs)

    system_prompt_template = """You are an expert Enterprise Financial Analyst specializing in Walmart's corporate performance and annual reports.

Answer the user's question accurately, professionally, and strictly using ONLY the provided context below.

Instructions:
1. Base your answer ONLY on the provided context. If the context does not contain enough information to answer, state clearly: "Based on the provided Walmart 2025 report context, I cannot answer this question."
2. Do NOT hallucinate, infer, or bring in external financial facts outside the text.
3. Present numerical figures, metrics, and percentages clearly.
4. Structure your response using markdown with bullet points for scannability.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {query}

Analytical Answer:"""

    prompt = PromptTemplate(
        template=system_prompt_template,
        input_variables=["context", "query"],
    )

    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    rag_chain = prompt | llm | StrOutputParser()
    raw = rag_chain.invoke({"context": formatted_context, "query": query})
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw
