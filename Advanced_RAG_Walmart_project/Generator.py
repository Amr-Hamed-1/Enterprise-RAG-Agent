import os
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved document chunks into a clean context string."""
    formatted_chunks = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Walmart Annual Report 2025")
        page = doc.metadata.get("page", "N/A")
        formatted_chunks.append(
            f"--- CHUNK {idx} (Source: {source} | Page: {page}) ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def generate_rag_response(query: str, context_docs: List[Document]) -> str:
    """Generates a strictly grounded financial response using Llama 3.1 on Groq."""
    print("🤖 [Phase 5] Generating Response with Groq Llama 3.1...")
    
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
        input_variables=["context", "query"]
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain.invoke({
        "context": formatted_context,
        "query": query
    }) 