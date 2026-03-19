# app.py — RAG Document Intelligence System
# Author: Divya VJ
# Stack: LangChain · ChromaDB · Groq Llama-3 · Gemini · RAGAS · Streamlit

import streamlit as st
import os
import tempfile

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # on Hugging Face, secrets are already in environment

# ── Auto-create database directory ────────────────────────────────────────
os.makedirs("chroma_db", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0E1117; }
[data-testid="stSidebar"] {
    background-color: #0D2B5E;
    border-right: 2px solid #1A56A8;
}
[data-testid="stSidebar"] * { color: #DCE8FA !important; }
h1 { color: #1A56A8 !important;
     border-bottom: 2px solid #1A56A8;
     padding-bottom: 8px; }
h2, h3 { color: #2A6ED4 !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A56A8, #0D2B5E);
    border-radius: 10px;
    padding: 10px 16px;
    border: 1px solid #2A5FC8;
}
[data-testid="metric-container"] label {
    color: #A0C4FF !important;
    font-size: 12px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
.stButton button {
    background-color: #1A56A8;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 24px;
    font-weight: 600;
}
.stButton button:hover {
    background-color: #0D2B5E;
    color: white;
}
.stTextInput input {
    background-color: #1A1A2E;
    color: white;
    border: 1px solid #1A56A8;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("🔍 RAG Document QA")
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Upload any PDF document")
st.sidebar.markdown("2. System indexes it with AI embeddings")
st.sidebar.markdown("3. Ask questions in natural language")
st.sidebar.markdown("4. Get answers with source citations")
st.sidebar.markdown("5. See RAGAS quality score")
st.sidebar.markdown("---")

llm_choice = st.sidebar.selectbox(
    "Choose LLM:",
    ["groq (Llama-3 — Free)", "gemini (Gemini Flash — Free)"]
)
llm_key = "groq" if "groq" in llm_choice else "gemini"

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("LangChain · ChromaDB")
st.sidebar.markdown("Sentence-Transformers")
st.sidebar.markdown("Groq Llama-3 · Gemini")
st.sidebar.markdown("RAGAS Evaluation")

# ── Main Page ─────────────────────────────────────────────────────────────
st.title("🔍 RAG Document Intelligence System")
st.markdown(
    "*Upload any PDF. Ask questions in natural language. "
    "Get grounded answers with source citations and quality scores.*"
)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD DOCUMENT
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📄 Step 1: Upload Your Document")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    help="Upload any PDF — research paper, report, contract, notes"
)

if uploaded_file is not None:

    # Show file info
    col1, col2, col3 = st.columns(3)
    col1.metric("File Name", uploaded_file.name)
    col2.metric("File Size", f"{uploaded_file.size // 1024} KB")
    col3.metric("File Type", "PDF")

    # Index button
    if st.button("🚀 Index Document — Start AI Processing"):

        with st.spinner(
            "Loading document, creating embeddings, "
            "building vector store... (30-60 seconds)"
        ):
            try:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Import and run pipeline
                from ingest import (
                    load_document,
                    split_documents,
                    create_vector_store
                )

                pages  = load_document(tmp_path)
                chunks = split_documents(pages)

                # Use unique collection per document
                collection_name = uploaded_file.name.replace(
                    ".pdf", ""
                ).replace(" ", "_")[:50]

                vs, _ = create_vector_store(
                    chunks,
                    persist_dir=f"chroma_db/{collection_name}"
                )

                # Store in session state
                st.session_state["vector_store"]  = vs
                st.session_state["doc_name"]      = uploaded_file.name
                st.session_state["num_pages"]     = len(pages)
                st.session_state["num_chunks"]    = len(chunks)
                st.session_state["collection"]    = collection_name

                st.success(
                    f"✅ Document indexed successfully! "
                    f"{len(pages)} pages → {len(chunks)} chunks → "
                    f"ready for questions."
                )

            except Exception as e:
                st.error(f"❌ Error indexing document: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — ASK QUESTIONS
# ══════════════════════════════════════════════════════════════════════════
if "vector_store" in st.session_state:

    st.markdown("---")
    st.subheader("💬 Step 2: Ask Questions About Your Document")

    # Document stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Document",  st.session_state["doc_name"])
    c2.metric("Pages",     st.session_state["num_pages"])
    c3.metric("Chunks",    st.session_state["num_chunks"])
    c4.metric("LLM",       llm_key.upper())

    # Question input
    question = st.text_input(
        "Ask a question about your document:",
        placeholder="e.g. What is the main contribution of this paper?",
    )

    if st.button("🔍 Get Answer") and question:

        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                from retriever import build_rag_chain, ask_question

                chain_tuple = build_rag_chain(
                     st.session_state["vector_store"],
                     llm_choice=llm_key
                )
                result = ask_question(chain_tuple, question)

                # Store result in session state
                st.session_state["last_result"]   = result
                st.session_state["last_question"]  = question

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")

    # ── Display Results ────────────────────────────────────────────────────
    if "last_result" in st.session_state:
        result   = st.session_state["last_result"]
        question = st.session_state["last_question"]

        # ── Answer Box ────────────────────────────────────────────────────
        st.markdown("### 💡 Answer")
        st.markdown(f"""
        <div style="background:#0A4A25; border-left:4px solid #1A7A42;
                    padding:16px; border-radius:8px; color:#E0FFE0;
                    font-size:15px; line-height:1.7;">
        {result['answer']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RAGAS Score ────────────────────────────────────────────────────
        st.markdown("### 📊 Answer Quality — RAGAS Faithfulness Score")
        st.markdown(
            "*Measures whether every statement in the answer "
            "is supported by the retrieved document chunks.*"
        )

        with st.spinner("Evaluating answer quality with RAGAS..."):
            try:
                from utils.evaluator import evaluate_faithfulness
                contexts = [s["content"] for s in result["sources"]]
                score    = evaluate_faithfulness(
                    question, result["answer"], contexts
                )
            except Exception as e:
                score = None
                st.warning(f"Evaluation unavailable: {e}")

        if score is not None and not (
            isinstance(score, float) and score != score
        ):  # check for NaN
            if score >= 0.7:
                color = "#1A7A42"
                label = "HIGH QUALITY"
                emoji = "✅"
            elif score >= 0.4:
                color = "#C04A00"
                label = "MEDIUM QUALITY"
                emoji = "⚠️"
            else:
                color = "#7B0000"
                label = "LOW QUALITY"
                emoji = "❌"

            st.markdown(f"""
            <div style="background:#1A1A2E; border:2px solid {color};
                        padding:20px; border-radius:10px;
                        text-align:center; margin:10px 0;">
                <span style="font-size:42px; font-weight:bold;
                             color:{color};">{score:.0%}</span>
                <br>
                <span style="color:#aaa; font-size:13px;">
                    {emoji} RAGAS Faithfulness Score — {label}
                </span>
                <br><br>
                <span style="color:#888; font-size:11px;">
                    {score:.0%} of answer statements are grounded
                    in the retrieved document chunks
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(
                "📊 Quality score unavailable for this response. "
                "The answer may still be accurate."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Source Chunks ──────────────────────────────────────────────────
        st.markdown("### 📖 Source Chunks Used To Generate This Answer")
        st.markdown(
            "*These are the exact passages retrieved from your document. "
            "The answer is based only on these passages.*"
        )

        for i, src in enumerate(result["sources"]):
            with st.expander(
                f"📄 Source {i+1} — Page {src['page']} "
                f"of {st.session_state['doc_name']}"
            ):
                st.markdown(f"""
                <div style="background:#1A1A2E; padding:14px;
                            border-radius:6px;
                            border-left:3px solid #1A56A8;
                            color:#CDD6F4; font-size:13px;
                            line-height:1.7; font-family:monospace;">
                {src['content']}
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:12px; padding:8px'>
    RAG Document Intelligence · LangChain · ChromaDB ·
    Groq Llama-3 · Google Gemini · RAGAS · Built by Divya VJ
</div>
""", unsafe_allow_html=True)