from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────
# This is the exact text sent to Llama-3 for every question
# {context} gets replaced with retrieved chunks
# {question} gets replaced with user question
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
based ONLY on the provided document context below.

Rules:
- Use ONLY information from the context provided
- Do NOT use any outside knowledge
- If the answer is not in the context, say exactly:
  "I could not find information about this in the document."
- Keep your answer clear and concise
- Always mention which part of the document supports your answer

Context from document:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# ── BUILD RAG CHAIN ───────────────────────────────────────────────────────
def build_rag_chain(vector_store, llm_choice="groq"):
    """
    Build complete RAG chain.
    Connects: ChromaDB retriever → Prompt → LLM → Answer

    Args:
        vector_store: ChromaDB vector store object
        llm_choice: "groq" for Llama-3, "gemini" for Gemini Flash

    Returns: RetrievalQA chain
    """
    # ── Choose LLM ────────────────────────────────────────────────────────
    if llm_choice == "groq":
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )
        print(f"Using LLM: Groq Llama-3 8B")

    elif llm_choice == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print(f"Using LLM: Google Gemini 1.5 Flash")

    else:
        raise ValueError(f"Unknown LLM choice: {llm_choice}. Use 'groq' or 'gemini'")

    # ── Build Chain ───────────────────────────────────────────────────────
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain


# ── ASK QUESTION ──────────────────────────────────────────────────────────
def ask_question(chain, question):
    """
    Ask a question and return answer with source chunks.

    Args:
        chain: RetrievalQA chain from build_rag_chain()
        question: user question string

    Returns:
        dict with keys:
          - answer: generated answer string
          - sources: list of dicts with page, content, source
    """
    print(f"\nProcessing question: {question}")
    print(f"Retrieving relevant chunks...")

    result = chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    # Extract source information from retrieved chunks
    sources = []
    for doc in source_docs:
        sources.append({
            "page":    doc.metadata.get("page_label", str(doc.metadata.get("page", 0) + 1)),
            "content": doc.page_content,
            "source":  doc.metadata.get("source", "unknown")
        })

    return {
        "answer":  answer,
        "sources": sources
    }


# ── TEST BLOCK ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingest import load_vector_store

    print("="*50)
    print("RAG CHAIN TEST")
    print("="*50)

    # Load existing vector store
    print("\nStep 1: Loading vector store...")
    vs, _ = load_vector_store()

    # Build chain
    print("\nStep 2: Building RAG chain...")
    chain = build_rag_chain(vs, llm_choice="groq")

    # Test questions
    questions = [
        "What is the main contribution of this paper?",
        "What is multi-head attention and how does it work?",
        "What optimizer was used and what were the training details?",
        "What BLEU score did the Transformer achieve?"
    ]

    for q in questions:
        print(f"\n{'='*50}")
        result = ask_question(chain, q)

        print(f"\nQUESTION: {q}")
        print(f"\nANSWER: {result['answer']}")
        print(f"\nSOURCES USED:")
        for i, src in enumerate(result['sources']):
            print(f"  Source {i+1}: Page {src['page']}")
            print(f"  Preview: {src['content'][:100]}...")
        print()