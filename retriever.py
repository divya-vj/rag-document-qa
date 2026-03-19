# retriever.py — Modern LangChain approach (no RetrievalQA)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# ── Prompt Template ───────────────────────────────────────────────────────
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


# ── Format retrieved docs ─────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── Build RAG Chain ───────────────────────────────────────────────────────
def build_rag_chain(vector_store, llm_choice="groq"):
    """Build complete RAG chain using modern LCEL approach."""

    if llm_choice == "groq":
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2048,
            api_key=os.getenv("GROQ_API_KEY")
        )
        print(f"Using LLM: Groq Llama-3.1 8B")

    elif llm_choice == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print(f"Using LLM: Google Gemini 1.5 Flash")

    else:
        raise ValueError(f"Unknown LLM: {llm_choice}")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Modern LCEL chain
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── Ask Question ──────────────────────────────────────────────────────────
def ask_question(chain_and_retriever, question):
    """Ask question and return answer with source chunks."""
    chain, retriever = chain_and_retriever

    print(f"\nProcessing: {question}")

    # Get answer
    answer = chain.invoke(question)

    # Get source documents separately
    source_docs = retriever.invoke(question)

    sources = []
    for doc in source_docs:
        sources.append({
            "page":    doc.metadata.get(
                "page_label",
                str(doc.metadata.get("page", 0) + 1)
            ),
            "content": doc.page_content,
            "source":  doc.metadata.get("source", "unknown")
        })

    return {"answer": answer, "sources": sources}


# ── Test Block ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingest import load_vector_store

    print("="*50)
    print("RAG CHAIN TEST — Modern LCEL")
    print("="*50)

    print("\nLoading vector store...")
    vs, _ = load_vector_store()

    print("Building RAG chain...")
    chain_tuple = build_rag_chain(vs, llm_choice="groq")

    questions = [
        "What is the main contribution of this paper?",
        "What is multi-head attention and how does it work?",
        "What optimizer was used for training?",
    ]

    for q in questions:
        print(f"\n{'='*50}")
        result = ask_question(chain_tuple, q)
        print(f"QUESTION: {q}")
        print(f"ANSWER: {result['answer']}")
        print(f"SOURCES: Pages {[s['page'] for s in result['sources']]}")