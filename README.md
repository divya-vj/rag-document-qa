# 🔍 RAG Document Intelligence System

> Upload any PDF. Ask questions in plain English. Get answers grounded in the document — not hallucinated — with source citations and automated quality scores.

[

![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue?style=for-the-badge)

](https://huggingface.co/spaces/Divyavj1/rag-document-qa)
[

![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)

](https://python.org)
[

![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-orange?style=for-the-badge)

](https://langchain.com)

---

## The Problem This Solves

LLMs like GPT or Llama are trained on public data. They cannot answer questions about your private documents — internal reports, research papers, legal contracts.

This system solves that by building a RAG pipeline that:
- Indexes any PDF into a local vector database
- Retrieves only the relevant sections when you ask a question
- Generates an answer grounded in those sections, not in training data
- Shows exactly which chunks were used and how faithful the answer is

---

## Live Demo

**[→ Try it on Hugging Face Spaces](https://huggingface.co/spaces/Divyavj1/rag-document-qa)**

Tested on: *Attention Is All You Need* (Vaswani et al., 2017)

| Question Asked | RAGAS Faithfulness | Quality |
|---|---|---|
| What is the main contribution? | 100% | HIGH |
| What is multi-head attention? | 92% | HIGH |
| What optimizer was used? | 88% | HIGH |
| What BLEU score was achieved? | 85% | HIGH |

---

## How It Works

**Indexing phase — runs once when you upload a PDF:**

1. PDF is loaded and text is extracted with page-level metadata
2. Text is split into 1000-character chunks with 200-character overlap
3. Each chunk is converted into a 384-dimensional vector using `all-MiniLM-L6-v2`
4. Vectors are stored in a local ChromaDB collection

**Query phase — runs every time you ask a question:**

1. Your question is embedded using the same model
2. Top-4 most relevant chunks are retrieved by cosine similarity
3. Chunks + question are injected into a Llama-3 prompt
4. Answer is generated strictly from the retrieved context
5. RAGAS faithfulness score is computed to verify answer quality

---

## Tech Stack

| Component | Technology | Why This Choice |
|---|---|---|
| RAG pipeline | LangChain LCEL | Standard production pattern for RAG |
| Vector DB | ChromaDB | Runs locally, zero config, persists to disk |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Free, private, no data sent externally |
| LLM (primary) | Groq Llama-3.1 | Free tier, sub-second responses via custom LPU |
| LLM (alternate) | Google Gemini Flash | Free alternative, switchable from UI |
| Evaluation | RAGAS faithfulness | Quantifies answer grounding, not just fluency |
| UI | Streamlit | Rapid deployment, clean interface |
| Deployment | Hugging Face Spaces (Docker) | Free hosting, always-on |

---

## What Makes This Different From Generic RAG Projects

Most RAG demos stop at "it returned an answer." This one goes further:

- **Explainability layer** — every answer shows the exact document chunks used to generate it, with page numbers
- **RAGAS evaluation** — automated faithfulness scoring tells you whether the answer is grounded in the document or not
- **LLM switching** — Groq and Gemini are both available, switchable from the sidebar without any code change
- **Per-document collections** — each uploaded PDF gets its own ChromaDB collection

---

## Project Structure

```
rag-document-qa/
├── app.py            # Streamlit UI — upload, index, QA, RAGAS score display
├── ingest.py         # PDF loading → chunking → embedding → ChromaDB storage
├── retriever.py      # LCEL chain — semantic retrieval + answer generation
├── utils/
│   ├── evaluator.py  # RAGAS faithfulness evaluation
│   └── __init__.py
├── Dockerfile        # HF Spaces deployment config
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/divya-vj/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

Run:

```bash
streamlit run app.py
```

Free API keys:
- Groq → [console.groq.com](https://console.groq.com)
- Google → [aistudio.google.com](https://aistudio.google.com)

---


## Contact

**Divya VJ**

- LinkedIn: [linkedin.com/in/divyavj123](https://linkedin.com/in/divyavj123)
- GitHub: [github.com/divya-vj](https://github.com/divya-vj)
- Email: divya.aiml123@gmail.com
