# Enterprise RAG Agent

An end-to-end Enterprise Retrieval-Augmented Generation (RAG) agent built with **LangChain**, **Groq LLM (`llama-3.1-8b-instant`)**, **HuggingFace Embeddings**, and **FAISS Vector Database**. The agent ingests PDF reports (such as annual financial reports) and provides context-aware, accurate answers to complex enterprise queries.

---

## 🌟 Key Features

- **Semantic Chunking Strategy**: Dynamically splits loaded documents into contextually coherent chunks based on sentence embedding similarity (`SemanticChunker`), replacing traditional fixed-size character splitters.
- **High-Performance LLM**: Powered by Groq's `llama-3.1-8b-instant` model for low-latency and precise responses.
- **Dense Vector Search**: Embeds document chunks using `BAAI/bge-small-en-v1.5` HuggingFace model and indexes them with **FAISS**.
- **Automated Document Ingestion**: Ingests all PDF files placed inside the `data/` directory automatically.

---

## 🏗️ Architecture Pipeline

1. **Document Loading**: `PyPDFDirectoryLoader` reads all PDF files in the `data/` directory.
2. **Model Configuration**:
   - **Embedding Model**: `HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")`
   - **LLM**: `ChatGroq(model="llama-3.1-8b-instant")`
3. **Semantic Chunking**: `SemanticChunker` computes semantic similarity across sentences to split documents at logical boundaries (95th percentile breakpoint threshold).
4. **Vector Database**: `FAISS` indexes chunks into an in-memory vector space and exposes a similarity retriever.
5. **Retrieval & QA Chain**: Connects `create_stuff_documents_chain` and `create_retrieval_chain` with a custom RAG prompt template to answer questions strictly using retrieved context.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Groq API Key (get one from [Groq Console](https://console.groq.com/))

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Amr-Hamed-1/Enterprise-RAG-Agent.git
   cd Enterprise-RAG-Agent
   ```

2. **Set Up Virtual Environment**:
   ```bash
   conda create -n enterprise_RAG python=3.10 -y
   conda activate enterprise_RAG
   ```

3. **Install Dependencies**:
   ```bash
   pip install langchain-community langchain-experimental langchain-huggingface langchain-groq langchain-core langchain-classic faiss-cpu sentence-transformers pypdf python-dotenv
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## 📁 Project Structure

```text
Enterprise-RAG-Agent/
├── data/                         # Directory containing PDF documents (e.g., Walmart Annual Report 2025.pdf)
├── .env                          # API keys and environment configuration
├── app.py                        # Main RAG agent execution script
└── README.md                     # Project documentation
```

---

## 🎯 Usage

1. Place your target PDF documents inside the `data/` folder.
2. Run the application script:
   ```bash
   python app.py
   ```

### Sample Output

```text
1. Configuring Models...
2. Loading and Chunking PDF...
-> Splitted into 243 chunks.
3. Creating Vector Database...
4. Setting up the QA Chain...

--- Answer ---
According to the report, the total net sales for the year 2025 was $674,538 million and for the year 2024 was $642,637 million. The main drivers of this growth were:

1. Strong positive comparable sales across the company's U.S. segments and international markets...
```