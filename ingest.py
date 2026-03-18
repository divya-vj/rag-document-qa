# ingest.py
# Purpose: Load PDF → Split into chunks → Embed → Store in ChromaDB
# We build this file step by step across Days 1, 2, and 3

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# ── STEP 1: Load PDF ──────────────────────────────────────────────────────
def load_document(file_path):
    """
    Load a PDF file and return a list of Document objects.
    One Document per page.
    Each Document has:
      - page_content: the extracted text from that page
      - metadata: dictionary with source path and page number
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    print(f"\n{'='*50}")
    print(f"DOCUMENT LOADED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"File: {file_path}")
    print(f"Total pages: {len(pages)}")
    print(f"\nFirst page preview:")
    print(f"{pages[0].page_content[:300]}")
    print(f"\nFirst page metadata:")
    print(f"{pages[0].metadata}")
    print(f"\nCharacter count per page:")
    for i, page in enumerate(pages):
        print(f"  Page {i+1}: {len(page.page_content)} characters")

    return pages

# ── STEP 2: Split into chunks ─────────────────────────────────────────────
def split_documents(pages):
    """
    Split Document pages into smaller overlapping chunks.
    Why: LLMs have context window limits.
    A 15-page PDF cannot fit in one prompt.
    Chunking breaks it into manageable pieces.
    Overlap ensures no information is lost at boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # each chunk is maximum 1000 characters
        chunk_overlap=200,     # last 200 chars of chunk N repeated at start of chunk N+1
        separators=[
            "\n\n",            # try paragraph break first
            "\n",              # then line break
            ". ",              # then sentence end
            " ",               # then word break
            ""                 # last resort: character split
        ],
        length_function=len,   # measure size by character count
    )

    chunks = splitter.split_documents(pages)

    print(f"\n{'='*50}")
    print(f"DOCUMENT SPLIT SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks)//len(chunks)} characters")
    print(f"Smallest chunk: {min(len(c.page_content) for c in chunks)} characters")
    print(f"Largest chunk: {max(len(c.page_content) for c in chunks)} characters")

    print(f"\n--- CHUNK 1 ---")
    print(f"Content: {chunks[0].page_content}")
    print(f"Metadata: {chunks[0].metadata}")

    print(f"\n--- CHUNK 2 ---")
    print(f"Content: {chunks[1].page_content[:200]}")
    print(f"Metadata: {chunks[1].metadata}")

    print(f"\n--- OVERLAP CHECK ---")
    print(f"Last 200 chars of chunk 1:")
    print(f"  [{chunks[0].page_content[-200:]}]")
    print(f"First 200 chars of chunk 2:")
    print(f"  [{chunks[1].page_content[:200]}]")
    print(f"(These should look similar - that is the overlap working)")

    return chunks

# ── STEP 3: Create embeddings and store in ChromaDB ───────────────────────
def create_vector_store(chunks, persist_dir="chroma_db"):
    """
    Convert each chunk into a vector embedding and store in ChromaDB.

    What happens here:
    1. Load the sentence transformer model (all-MiniLM-L6-v2)
    2. For each chunk, convert text to a 384-dimensional vector
    3. Store all vectors + original text + metadata in ChromaDB
    4. Save to disk so we do not re-embed every time app restarts

    Returns: ChromaDB vector store object
    """
    print(f"\n{'='*50}")
    print(f"CREATING VECTOR STORE")
    print(f"{'='*50}")
    print(f"Loading embedding model: all-MiniLM-L6-v2")
    print(f"(First run downloads ~90MB — please wait)")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Embedding model loaded successfully")
    print(f"Embedding {len(chunks)} chunks into ChromaDB...")
    print(f"This takes 30-60 seconds...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    count = vector_store._collection.count()
    print(f"\nVector store created successfully")
    print(f"Total vectors stored: {count}")
    print(f"Saved to disk at: {persist_dir}/")

    # Test: show what a single embedding looks like
    sample_text = chunks[0].page_content[:100]
    sample_vector = embeddings.embed_query(sample_text)
    print(f"\nSample embedding info:")
    print(f"  Text: {sample_text[:50]}...")
    print(f"  Vector dimensions: {len(sample_vector)}")
    print(f"  First 5 values: {[round(v, 4) for v in sample_vector[:5]]}")

    return vector_store, embeddings


# ── STEP 3b: Load existing vector store from disk ─────────────────────────
def load_vector_store(persist_dir="chroma_db"):
    """
    Load an already-created ChromaDB from disk.
    Use this instead of create_vector_store on subsequent runs.
    Much faster — no re-embedding needed.
    """
    print(f"Loading existing vector store from {persist_dir}/")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    count = vector_store._collection.count()
    print(f"Loaded vector store with {count} vectors")
    return vector_store, embeddings


# ── TEST BLOCK ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # ── Days 1 and 2: Load and Split ──────────────────────────────────────
    pages  = load_document("docs/attention.pdf")
    chunks = split_documents(pages)

    # ── Day 3: Embed and Store ────────────────────────────────────────────
    # Check if vector store already exists
    if os.path.exists("chroma_db") and len(os.listdir("chroma_db")) > 0:
        print("\nVector store already exists — loading from disk")
        vs, embeddings = load_vector_store()
    else:
        print("\nCreating new vector store...")
        vs, embeddings = create_vector_store(chunks)

    # ── Test: Semantic Search ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"SEMANTIC SEARCH TEST")
    print(f"{'='*50}")

    test_questions = [
        "What is the attention mechanism?",
        "What optimizer was used for training?",
        "How many layers does the encoder have?"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print(f"Top 3 retrieved chunks:")

        results = vs.similarity_search_with_score(question, k=3)

        for i, (doc, score) in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"  Distance score: {score:.4f} (lower = more similar)")
            print(f"  Page: {doc.metadata.get('page_label', '?')}")
            print(f"  Content preview: {doc.page_content[:150]}...")