from dotenv import load_dotenv
import os

load_dotenv()


def evaluate_faithfulness(question, answer, contexts):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset

        data = {
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        eval_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=2048,
            api_key=os.getenv("GROQ_API_KEY")
        )

        eval_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("Running RAGAS faithfulness evaluation...")
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=LangchainLLMWrapper(eval_llm),
            embeddings=LangchainEmbeddingsWrapper(eval_embeddings)
        )

        faith_val = result["faithfulness"]
        if isinstance(faith_val, list):
            faith_val = faith_val[0]
        score = round(float(faith_val), 2)
        print(f"Faithfulness score: {score}")
        return score

    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        return None


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest import load_vector_store
    from retriever import build_rag_chain, ask_question

    print("="*50)
    print("RAGAS EVALUATION TEST")
    print("="*50)

    print("\nLoading vector store...")
    vs, _ = load_vector_store()

    print("Building RAG chain...")
    chain = build_rag_chain(vs, llm_choice="groq")

    question = "What is multi-head attention and how does it work?"
    print(f"\nQuestion: {question}")

    result = ask_question(chain, question)
    print(f"\nAnswer: {result['answer'][:200]}...")

    print("\nEvaluating answer faithfulness...")
    contexts = [src["content"] for src in result["sources"]]
    score = evaluate_faithfulness(question, result["answer"], contexts)

    if score is not None:
        print(f"\n{'='*50}")
        print(f"RAGAS FAITHFULNESS SCORE: {score:.0%}")
        if score >= 0.7:
            print(f"Quality: HIGH — answer well grounded in document")
        elif score >= 0.4:
            print(f"Quality: MEDIUM — some statements may not be grounded")
        else:
            print(f"Quality: LOW — answer may contain hallucinations")
        print(f"{'='*50}")
    else:
        print("Evaluation could not be completed")