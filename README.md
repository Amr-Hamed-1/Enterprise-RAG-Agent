<div align="center">

# 🏢 Enterprise RAG Agent

**End-to-end Retrieval-Augmented Generation over the Walmart 2025 Annual Report**
A production-style RAG system with hybrid retrieval, cross-encoder reranking, LLM query rewriting, grounded generation, and LLM-judged evaluation.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-3D5A80?logo=qdrant&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?logo=groq&logoColor=white)
![DeepEval](https://img.shields.io/badge/DeepEval-4%20metrics-6D28D9)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## 📌 Overview

This project is a complete, deployable **Enterprise RAG Agent** built on a real-world use case: answering financial questions about **Walmart's 2025 Annual Report** (97 pages, PDF) with *grounded, cited, non-hallucinated* answers.

It started as 7 Jupyter notebooks walking through the full RAG lifecycle, then matured into a **FastAPI microservice** with a **Streamlit chat UI**, a **hybrid retrieval pipeline**, and an **LLM-judged evaluation suite**. A **Naive baseline** is included so the advanced system can be measured against a simple one — the way real ML engineering teams justify architecture decisions.

**The star of the repo:** a clean separation between a "learn the concepts" journey (notebooks `01 → 07`) and a production-grade implementation you can actually deploy.

---

## ✨ Key Features

| Capability | Implementation |
|---|---|
| 🧠 **Hybrid Retrieval** | Dense (Qdrant) **+** sparse (BM25) fused via `EnsembleRetriever` (50/50) |
| ⚡ **Cross-Encoder Reranking** | Local **FlashRank** reranks candidates → only top-N reach the LLM |
| 🔄 **Query Rewriting** | `llama-3.3-70b-versatile` rewrites questions into search-optimized queries (`temperature=0`) |
| 🛡️ **Anti-Hallucination** | Strict prompt that grounds answers *only* in retrieved context |
| 🌐 **Vector DB as a Service** | Qdrant Cloud (managed), embedding via `all-MiniLM-L6-v2` |
| 🔌 **REST API** | FastAPI with rate limiting, request validation, and structured JSON (answer + sources + latency breakdown) |
| 💬 **Chat UI** | Streamlit interface with expandable source chunks and timing telemetry |
| 🧪 **LLM-Judged Evaluation** | DeepEval — Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall |
| 📊 **Diagnostics** | Golden-set retriever/generator checker (`diagnose.py`) + parameter sweep (`tune_retrieval.py`) |
| 🔒 **Security-first** | Keys via `.env` only, no hardcoded secrets, per-IP sliding-window rate limiting |

---

## 🏗️ Architecture

```
            ┌─────────────────────────────────────────────────────────┐
            │   Clients: Streamlit UI  ·  CLI (main.py)  ·  curl      │
            └──────────────────────────┬──────────────────────────────┘
                                       │ HTTP
                                       ▼
                         ┌─────────────────────────┐
                         │   FastAPI Gateway       │
                         │  rate-limit · validation│
                         └───────────┬─────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  Query Rewriter (Llama 3.3 70B)│
                    │  temperature = 0 (deterministic)│
                    └───────────────────┬────────────┘
                                        │ rewritten query
                                        ▼
                       ┌────────────────────────────────┐
                       │       HYBRID RETRIEVAL         │
                       │   ┌───────────┐  ┌───────────┐ │
                       │   │ Dense     │  │ Sparse    │ │
                       │   │ Qdrant    │  │ BM25      │ │
                       │   └─────┬─────┘  └─────┬─────┘ │
                       │         └──────┬────────┘      │
                       │    EnsembleRetriever (0.5/0.5) │
                       └───────────────────┬────────────┘
                                           │ top-k candidates
                                           ▼
                       ┌────────────────────────────────┐
                       │   FlashRank Cross-Encoder      │
                       │        rerank → top-N          │
                       └───────────────────┬────────────┘
                                           │ top-N grounded chunks
                                           ▼
                    ┌────────────────────────────────┐
                    │  Generator (Llama 3.1 8B)      │
                    │  answers ONLY from context     │
                    └───────────────────┬────────────┘
                                        ▼
              Answer  +  Cited Sources  +  Latency breakdown
```

### The RAG Pipeline (phase by phase)

| # | Phase | What it does |
|---|-------|--------------|
| 1 | **Document Loading** | `PyMuPDFLoader` parses the 97-page PDF page-by-page with page numbers preserved in metadata |
| 2 | **Chunking** | `RecursiveCharacterTextSplitter` → **650 chunks** (800 chars, 150 overlap) |
| 3 | **Embeddings & Vector DB** | `all-MiniLM-L6-v2` embeddings stored in a **Qdrant Cloud** collection |
| 4 | **Hybrid Retrieval + Reranking** | BM25 + dense ensemble, then **FlashRank** cross-encoder reranking |
| 5 | **Query Transformation** | LLM rewrites the user query for better retrieval (deterministic via `temp=0`) |
| 6 | **Generation** | `llama-3.1-8b-instant` generates a grounded, markdown answer with anti-hallucination guardrails |
| 7 | **Evaluation** | DeepEval measures **Faithfulness · Answer Relevancy · Contextual Precision · Contextual Recall** using a 70B judge |

---

## ⚖️ Naive vs. Advanced — the honest comparison

One repo, two systems, same question — that's what turns a demo into a **portfolio piece**.

| Dimension | 🐣 Naive (`simple_Naive_RAG_project`) | 🚀 Advanced (`Advanced_RAG_Walmart_project`) |
|---|---|---|
| Chunking | Semantic Chunker | Recursive splitter (800/150) |
| Vector DB | FAISS (in-memory) | Qdrant Cloud (managed) |
| Embeddings | `bge-small-en-v1.5` | `all-MiniLM-L6-v2` |
| Retrieval | Dense only | **Hybrid** (dense + BM25) |
| Reranking | — | **FlashRank** cross-encoder |
| Query rewriting | — | **LLM rewrite** (`temp=0`) |
| Guardrails | basic prompt | strict grounding + fallback refusal |
| Interface | single script | **FastAPI + Streamlit + CLI** |
| Evaluation | — | **DeepEval (4 metrics, 70B judge)** |
| Extras | — | rate limiting, latency telemetry, diagnosis tooling |

---

## 📁 Project Structure

```text
Enterprise-RAG-Agent/
├── 01_data_loading/                    # 📓 Notebook phases — the learning journey
├── 02_data_chunking/
├── 03_VectorDB&Embedings/
├── 04_Advanced_retrieval&Reranking/
├── 05_Query_Transformation/
├── 06_Generation & Prompt Engineering/
├── 07_Evaluation & Metrics/
│
├── simple_Naive_RAG_project/           # 🐣 Naive baseline (FAISS + bge + single script)
│   └── Naive_RAG.py
│
├── Advanced_RAG_Walmart_project/       # 🚀 Production-grade implementation
│   ├── app/                            #    core package
│   │   ├── config.py                   #      paths + env + limits
│   │   ├── loader.py                   #      PDF → chunks → Qdrant
│   │   ├── retriever.py                #      Qdrant + query rewrite
│   │   ├── generator.py                #      grounded generation
│   │   ├── service.py                  #      RAGService orchestrator
│   │   ├── schemas.py                  #      Pydantic models
│   │   ├── api.py                      #      FastAPI app
│   │   └── evaluation.py               #      DeepEval benchmark
│   ├── ui/streamlit_app.py             #    chat interface
│   ├── scripts/                        #    diagnose.py · tune_retrieval.py
│   ├── tests/test_questions.json       #    golden ground-truth set
│   └── main.py                         #    CLI entry point
│
├── data/Walmart Annual Report 2025.pdf # 📄 Source corpus (11.3 MB)
├── requirements.txt
├── .env                                # 🔒 keys (git-ignored)
└── README.md
```

---

## 🚀 Getting Started

### 1 · Prerequisites

- **Python 3.10** (conda recommended)
- **Groq API key** — [console.groq.com](https://console.groq.com)
- **Qdrant Cloud** cluster + API key — [qdrant.tech](https://qdrant.tech) (free tier is enough)

### 2 · Environment

```bash
conda create -n enterprise_RAG python=3.10 -y
conda activate enterprise_RAG
pip install -r requirements.txt
```

Create `.env` in the repo root (or inside `Advanced_RAG_Walmart_project/`):

```env
GROQ_API_KEY=your_groq_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key

# Optional tuning
MAX_REQUESTS_PER_MINUTE=10
MAX_QUERY_LENGTH=500
```

### 3 · Index the PDF (one-time)

```bash
cd Advanced_RAG_Walmart_project
python -m app.loader        # parses, chunks, uploads 650 chunks to Qdrant
```

### 4 · Run it

```bash
# 🔌 REST API  →  http://127.0.0.1:8000/docs
uvicorn app.api:app --reload

# 💬 Chat UI   →  http://localhost:8501   (run in the env that has streamlit)
streamlit run ui/streamlit_app.py

# ⌨️ CLI
python main.py "What were Walmart's total revenues in fiscal year 2025?"
```

### 5 · Test & evaluate

```bash
# Golden-set diagnosis (retrieval hit-rate + grounded number check)
python scripts/diagnose.py 10 5

# Retrieval parameter sweep (k / top_n)
python scripts/tune_retrieval.py

# Official DeepEval benchmark (4 metrics, 70B judge)
python -m app.evaluation
```

---

## 🎯 Example

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How much were Sam\u0027s Club U.S. grocery net sales in fiscal year 2025?", "k": 10, "top_n": 5}'
```

**Response:**

```json
{
  "query": "How much were Sam's Club U.S. grocery net sales in fiscal year 2025?",
  "rewritten_query": "Sam's Club U.S. grocery net sales FY2025",
  "answer": "Sam's Club U.S. grocery net sales were **$59,976 million** in fiscal year 2025...",
  "sources": [
    {
      "content": "...grocery net sales were $59,976 million...",
      "page": 81,
      "source": "Walmart Annual Report 2025.pdf"
    }
  ],
  "metrics": {
    "num_chunks_indexed": 650,
    "query_rewrite_ms": 1135.1,
    "retrieval_ms": 8217.5,
    "generation_ms": 1154.2,
    "total_ms": 10506.8
  }
}
```

---

## 📊 Results & Evaluation

> **Honest numbers.** The golden set (`tests/test_questions.json`) was built by *verifying every expected answer against the actual PDF pages* — no made-up facts.

- **Corpus:** 97 pages → 650 chunks → Qdrant Cloud
- **Diagnosis (k=10, top_n=5):** **7 / 8 questions fully PASS** (correct number in answer **and** correct page retrieved)
- **Known limitation:** the 1 failing case is a *dense infographic* on the cover page (page 1) — a classic layout-extraction problem, not a retrieval bug. Great interview story. 🎯
- **Latency:** ≈ 1 s (rewrite) + 2–8 s (hybrid retrieval + rerank) + ≈ 1 s (generation)

**DeepEval** (Faithfulness · Answer Relevancy · Contextual Precision · Contextual Recall, judged by `llama-3.3-70b-versatile`) is fully wired up — run `python -m app.evaluation` to produce the official scores.

---

## 🛣️ Roadmap

- [x] Hybrid retrieval + FlashRank reranking
- [x] LLM query rewriting (`temp=0`)
- [x] FastAPI microservice + Streamlit UI
- [x] Rate limiting & request validation
- [x] DeepEval evaluation suite
- [ ] Dockerize API + UI (`docker-compose`)
- [ ] Naive-vs-Advanced DeepEval score comparison (same golden set)
- [ ] Deploy to Hugging Face Spaces
- [ ] Layout-aware parsing for infographic-heavy pages

---

## 🛡️ Security Notes

- All secrets live in `.env` (git-ignored) — **zero hardcoded keys**
- Per-IP sliding-window rate limiter (`MAX_REQUESTS_PER_MINUTE`, default 10) + max query length guard
- Generation prompt enforces grounding → reduces prompt-injection / hallucination surface

---

<div align="center">

Built with **LangChain · Qdrant · Groq · FastAPI · Streamlit · DeepEval**

</div>
