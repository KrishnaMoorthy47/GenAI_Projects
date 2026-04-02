# Krishna's Resume Ingestion — Google Colab
# ============================================================
# Run each cell in order.
# You only need: your resume PDF + 3 API keys.
# ============================================================

# ── CELL 1: Install dependencies ──────────────────────────────
# !pip install -q openai supabase pypdf

# ── CELL 2: Paste your keys here ──────────────────────────────
OPENAI_API_KEY      = "sk-..."          # platform.openai.com
SUPABASE_URL        = "https://xxxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJ..."         # Supabase → Settings → API → service_role

COLLECTION = "krishna-profile"
CHUNK_SIZE = 400        # tokens per chunk (approximate by words)
CHUNK_OVERLAP = 50      # words overlap between chunks

# ── CELL 3: Upload your resume PDF ────────────────────────────
# from google.colab import files
# uploaded = files.upload()   # pick your resume PDF
# RESUME_PATH = list(uploaded.keys())[0]
# print("Uploaded:", RESUME_PATH)

# For testing without upload, set path manually:
# RESUME_PATH = "resume.pdf"

# ── CELL 4: Extract text from PDF ────────────────────────────
from pypdf import PdfReader

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Returns list of (page_num, text) tuples."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    print(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages

# ── CELL 5: Split into chunks ─────────────────────────────────
def chunk_text(page_num: int, text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Split page text into word-based chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({
            "page_num": page_num,
            "chunk_index": idx,
            "content": chunk,
        })
        idx += 1
        start += chunk_size - overlap
    return chunks

def split_pages(pages: list[tuple[int, str]]) -> list[dict]:
    all_chunks = []
    for page_num, text in pages:
        all_chunks.extend(chunk_text(page_num, text, CHUNK_SIZE, CHUNK_OVERLAP))
    print(f"Split into {len(all_chunks)} chunks")
    return all_chunks

# ── CELL 6: Embed chunks ──────────────────────────────────────
from openai import OpenAI
import time

def embed_chunks(chunks: list[dict], openai_api_key: str) -> list[dict]:
    client = OpenAI(api_key=openai_api_key)
    texts = [c["content"] for c in chunks]

    # Batch in groups of 100 (OpenAI limit)
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        embeddings.extend([r.embedding for r in resp.data])
        print(f"  Embedded {min(i+100, len(texts))}/{len(texts)} chunks...")
        if i + 100 < len(texts):
            time.sleep(0.5)  # rate limit buffer

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    print(f"Embedding done — {len(chunks)} chunks ready")
    return chunks

# ── CELL 7: Insert into Supabase ─────────────────────────────
from supabase import create_client

def insert_chunks(
    chunks: list[dict],
    source_name: str,
    collection: str,
    supabase_url: str,
    supabase_key: str,
) -> None:
    sb = create_client(supabase_url, supabase_key)

    # Delete existing chunks for this source+collection (idempotent re-run)
    sb.table("chatbot_chunks").delete().eq("collection", collection).eq("source", source_name).execute()
    print(f"Cleared old chunks for source='{source_name}' collection='{collection}'")

    rows = [
        {
            "collection":   collection,
            "source":       source_name,
            "page_num":     c["page_num"],
            "chunk_index":  c["chunk_index"],
            "content":      c["content"],
            "embedding":    c["embedding"],
        }
        for c in chunks
    ]

    # Insert in batches of 50
    for i in range(0, len(rows), 50):
        batch = rows[i:i+50]
        sb.table("chatbot_chunks").insert(batch).execute()
        print(f"  Inserted {min(i+50, len(rows))}/{len(rows)} rows...")

    print(f"\nDone! {len(rows)} chunks stored in Supabase.")

# ── CELL 8: Run everything ────────────────────────────────────
import os

RESUME_PATH = "resume.pdf"   # change this to your uploaded filename

source_name = os.path.basename(RESUME_PATH)

pages  = extract_pages(RESUME_PATH)
chunks = split_pages(pages)
chunks = embed_chunks(chunks, OPENAI_API_KEY)
insert_chunks(chunks, source_name, COLLECTION, SUPABASE_URL, SUPABASE_SERVICE_KEY)
