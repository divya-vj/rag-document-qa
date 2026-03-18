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


# ── TEST BLOCK ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test Day 1: Load
    pages = load_document("docs/attention.pdf")

    # Test Day 2: Split
    chunks = split_documents(pages)

    # Extra analysis
    print(f"\n--- CHUNK DISTRIBUTION BY PAGE ---")
    page_chunk_count = {}
    for chunk in chunks:
        page = chunk.metadata.get('page_label', chunk.metadata.get('page', '?'))
        page_chunk_count[page] = page_chunk_count.get(page, 0) + 1
    for page, count in sorted(page_chunk_count.items(), key=lambda x: str(x[0])):
        print(f"  Page {page}: {count} chunks")