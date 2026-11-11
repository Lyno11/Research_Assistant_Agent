"""Diagnostic: shows exactly what the agent can read. Run from the project root:

    python check_docs.py

It uses no Gemini quota for the file listing. Add a question to also test
retrieval (that part does use a small amount of embedding quota):

    python check_docs.py "when is the deadline"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / "research_assistant" / ".env")

from research_assistant.loaders import SUPPORTED, read_any
from research_assistant.retriever import DOCS_DIR, INDEX_FILE, load_chunks

print(f"Docs folder: {DOCS_DIR}")
print(f"Supported types: {SUPPORTED}\n")

files = sorted(p for p in DOCS_DIR.glob("*") if p.is_file())
if not files:
    print("The folder is empty. Put your file in the folder shown above.")
    sys.exit(1)

for path in files:
    try:
        text = read_any(path)
    except Exception as error:
        print(f"  ERROR  {path.name}: {error}")
        continue
    if path.suffix.lower() not in SUPPORTED:
        print(f"  SKIP   {path.name}: unsupported type")
    elif not text.strip():
        print(f"  EMPTY  {path.name}: no text found (a scanned PDF needs OCR)")
    else:
        preview = " ".join(text.split())[:80]
        print(f"  OK     {path.name}: {len(text)} characters | {preview}...")

chunks = load_chunks()
print(f"\n{len(chunks)} passage(s) from {len({c['source'] for c in chunks})} file(s).")
print(f"Saved index: {'present' if INDEX_FILE.exists() else 'none yet'} ({INDEX_FILE})")

if len(sys.argv) > 1:
    from research_assistant.retriever import Retriever

    print(f"\nTop passages for: {sys.argv[1]}")
    for hit in Retriever().search(sys.argv[1]):
        print(f"  {hit['score']}  {hit['source']}: {' '.join(hit['passage'].split())[:100]}...")
