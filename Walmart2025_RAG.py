from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv


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

def main():
    pdf_path = "data/Walmart Annual Report 2025.pdf"
    docs = load_and_parse_walmart_docs(pdf_path)
    chunked_docs = chunking_docs(docs)

if __name__ == "__main__":
    main()
