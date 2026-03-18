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


# ── TEST BLOCK ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pages = load_document("docs/attention.pdf")
    print(f"\nTotal Document objects: {len(pages)}")
    print(f"\nPage 3 content preview:")
    print(pages[2].page_content[:400])
    print(f"\nPage 3 metadata: {pages[2].metadata}")