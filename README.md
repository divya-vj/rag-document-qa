---
title: RAG Document Intelligence System
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# 🔍 RAG Document Intelligence System

A production RAG system that makes any PDF queryable through natural language.
Upload a document, ask questions, get grounded answers with source citations
and automated quality scores.

**[🔗 Live Demo](https://huggingface.co/spaces/Divyavj1/rag-document-qa)**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [How RAG Works](#how-rag-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Contact](#contact)

---

## 🎯 Overview

LLMs cannot answer questions about your private documents. This system solves that.

Upload any PDF — research paper, legal contract, or report — ask anything in
plain English and get an answer grounded in that document, not hallucinated
from training data.

**What makes this different from other RAG projects:**
Every answer shows the exact document chunks retrieved, their page numbers,
and a RAGAS faithfulness score — so users can verify the answer is actually
supported by the document.

**Tested on:** Attention Is All You Need (Vaswani et al., 2017)

---

## 💡 Key Results

| Question | RAGAS Faithfulness | Quality |
|---|---|---|
| What is the main contribution? | 100% | HIGH |
| What is multi-head attention? | 92% | HIGH |
| What optimizer was used? | 88% | HIGH |
| What BLEU score was achieved? | 85% | HIGH |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| LangChain | RAG pipeline orchestration |
| ChromaDB | Vector database for semantic search |
| sentence-transformers | Free local embeddings (all-MiniLM-L6-v2) |
| Groq API | Free Llama-3.1 inference |
| Google Gemini | Alternative LLM option |
| RAGAS | Automated faithfulness evaluation |
| Streamlit | Interactive web interface |
| Hugging Face Spaces | Deployment |

**Why local embeddings over OpenAI?** Free, private, no data sent externally.
all-MiniLM-L6-v2 performs comparably to OpenAI embeddings on document retrieval.

**Why ChromaDB over Pinecone?** Zero configuration, runs locally, persists to disk.
No cloud signup, no usage limits, no cost.

**Why Groq over OpenAI?** Free tier with sub-second response times.
Custom LPU hardware makes Llama-3.1 faster than most paid APIs.

---

## 🔍 How RAG Works

**Indexing — runs once per document:**
1. Load PDF and extract text with page metadata
2. Split into 1000-character chunks with 200-character overlap
3. Embed each chunk into a 384-dimensional vector
4. Store vectors and metadata in ChromaDB

**Retrieval — runs on every question:**
1. Embed question using the same model
2. Find top-4 chunks by cosine similarity
3. Inject chunks and question into Llama-3 prompt
4. Generate grounded answer
5. Score with RAGAS faithfulness

---

## 📁 Project Structure

rag-document-qa/
├── app.py              # Streamlit UI — upload, index, QA, RAGAS score
├── ingest.py           # Load PDF → split → embed → store in ChromaDB
├── retriever.py        # LCEL chain — retrieve and generate answer
├── utils/
│   ├── evaluator.py    # RAGAS faithfulness evaluation
│   └── __init__.py
├── Dockerfile          # HF Spaces deployment
├── requirements.txt    # Python dependencies
└── README.md           # This file

---

## ⚙️ Installation

git clone https://github.com/divya-vj/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt

Add your API keys to a .env file:
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key

Run the app:
streamlit run app.py

Get free API keys at:
- Groq: console.groq.com
- Google: aistudio.google.com

---

## Contact

**Divya VJ**
- LinkedIn: linkedin.com/in/divyavj123
- GitHub: github.com/divya-vj
- Email: divya.aiml123@gmail.com

---

