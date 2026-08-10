import requests
import streamlit as st

st.set_page_config(page_title="Enterprise RAG Agent", page_icon="🏢", layout="wide")

API_BASE = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000", key="api_url")
k = st.sidebar.slider("k (candidates per retriever)", 5, 30, 10)
top_n = st.sidebar.slider("top_n (final chunks to LLM)", 1, 10, 5)
st.sidebar.divider()
if st.sidebar.button("🧹 Clear conversation"):
    st.session_state.messages = []
    st.rerun()
st.sidebar.caption("Backend: FastAPI · Qdrant + BM25 · FlashRank · Groq Llama")

st.title("📊 Enterprise RAG Agent")
st.caption("Ask questions about the Walmart 2025 Annual Report — grounded answers with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def ask_api(query: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/ask",
        json={"query": query, "k": k, "top_n": top_n},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            page = src.get("page", "N/A")
            source = src.get("source", "Unknown")
            st.markdown(f"**Chunk {i}** — page **{page}** · *{source}*")
            st.write(src["content"])
            st.divider()


if prompt := st.chat_input("Ask about the report..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Running the RAG pipeline..."):
                result = ask_api(prompt)

            st.markdown(result["answer"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🔍 Rewritten query:** {result['rewritten_query']}")
            with col2:
                m = result["metrics"]
                st.caption(
                    f"Total **{m['total_ms']} ms** "
                    f"(rewrite {m['query_rewrite_ms']} · retrieval {m['retrieval_ms']} · generation {m['generation_ms']})"
                )

            render_sources(result["sources"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Cannot reach the API at {API_BASE}. Start the backend first:\n\n"
                f"`uvicorn app.api:app --reload`"
            )
        except requests.exceptions.HTTPError as exc:
            st.error(f"❌ API error: {exc}")
        except Exception as exc:
            st.error(f"❌ Something went wrong: {exc}")
