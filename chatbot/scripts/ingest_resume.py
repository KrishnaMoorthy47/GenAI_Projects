"""
One-time ingestion script — upload resume/docs into the "krishna-profile" collection.

Usage:
    .venv/bin/python scripts/ingest_resume.py ~/resume.pdf [other_doc.pdf ...]
"""

import os
import sys
import time

import httpx

API_BASE = os.getenv("CHATBOT_API_BASE", "http://localhost:8002")
API_KEY = os.getenv("CHATBOT_API_KEY", "dev-secret")
COLLECTION = "krishna-profile"


def upload_file(path: str) -> None:
    filename = os.path.basename(path)
    print(f"Uploading: {filename} ...", end=" ", flush=True)
    start = time.time()

    try:
        with open(path, "rb") as f:
            resp = httpx.post(
                f"{API_BASE}/api/v1/upload",
                headers={"x-api-key": API_KEY},
                data={"collection": COLLECTION},
                files={"file": (filename, f)},
                timeout=120.0,
            )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start
        chunks = data.get("chunks_stored", data.get("chunk_count", "?"))
        print(f"done — {chunks} chunks stored ({elapsed:.1f}s)")
    except httpx.HTTPStatusError as e:
        print(f"FAILED (HTTP {e.response.status_code}: {e.response.text})")
    except FileNotFoundError:
        print(f"FAILED (file not found: {path})")
    except Exception as e:
        print(f"FAILED ({e})")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_resume.py <file1> [file2 ...]")
        sys.exit(1)

    files = sys.argv[1:]
    print(f"Ingesting {len(files)} file(s) into collection '{COLLECTION}'")
    print(f"Server: {API_BASE}\n")

    for path in files:
        upload_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
