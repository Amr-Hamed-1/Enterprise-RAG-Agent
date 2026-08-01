from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()




def load_and_parse_walmart_docs(pdf_path: str) -> List[Document]:
    """
    Phase 1: Load and parse PDF file into structured LangChain Document objects.
    Extracts text page-by-page with exact page numbers in metadata.
    """

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    print(f"✅ [Phase 1 Complete] Successfully parsed {len(docs)} pages from the PDF.\n")

    if docs:
        print("🔍 --- Sample Output (Page 1) ---")
        print(f"Content Preview: {docs[0].page_content[:200]}...")
        print(f"Metadata: {docs[0].metadata}")
        print("-----------------------------------\n")

    return docs




def chunking_docs(docs: List[Document]) -> List[Document]:
    """
    Phase 2: Split documents into smaller chunks for better retrieval performance.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    
    print(f"✅ [Phase 2 Complete] Split {len(docs)} documents into {len(chunked_docs)} chunks.\n")
    
    return chunked_docs    




def create_qdrant_vector_store(chunked_docs: List[Document], collection_name: str = "walmart_annual_report_2025")->QdrantVectorStore:

    """
    Phase 3: Create Qdrant vector store from chunked documents.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True},
    )

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY not found in environment variables") 

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        check_compatibility=False 
    )       

    vector_store = QdrantVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        api_key=qdrant_api_key,
        url=qdrant_url,
        collection_name=collection_name,
        force_recreate=True,
    )
    
    print(f"✅ [Phase 3 Complete] Created Qdrant vector store with {len(chunked_docs)} chunks.")
    
    return vector_store 





def main():
    pdf_path = "./data/Walmart Annual Report 2025.pdf"
    docs = load_and_parse_walmart_docs(pdf_path)
    chunked_docs = chunking_docs(docs)
    vector_store = create_qdrant_vector_store(chunked_docs)

if __name__ == "__main__":
    main()
