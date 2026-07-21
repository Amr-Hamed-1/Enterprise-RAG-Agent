import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

print("1. Configuring Models...")

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

#============================================

print("2. Loading and Chunking PDF...")

# Read all PDF files inside the data folder
loader = PyPDFDirectoryLoader("data")
docs = loader.load()

# Define semantic chunking configuration using embedding model
text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

# Perform text chunking on the documents
chunks = text_splitter.split_documents(docs)
print(f"-> Splitted into {len(chunks)} chunks.")

#============================================

print("3. Creating Vector Database...")

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

#============================================

print("4. Setting up the QA Chain...")

# Design the prompt template (guides the LLM output)
prompt = PromptTemplate.from_template("""
Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}
""")

# Connect all components into the retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Invoke the retrieval chain with the question
response = retrieval_chain.invoke({
    "input": "Based on the report, what were the total net sales or revenue for the year, and what were the main drivers of this growth?"
})

print("\n--- Answer ---")
print(response["answer"])