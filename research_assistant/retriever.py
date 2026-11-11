"""Retrieval half of RAG: load documents, split into chunks, embed, search.

Kept deliberately small so every line can be explained in the workshop.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .loaders import SUPPORTED, read_any

EMBEDDING_MODEL = "gemini-embedding-001"  # free tier; confirm at ai.google.dev/gemini-api/docs/embeddings
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
INDEX_FILE = Path(__file__).resolve().parent / ".index.json"
CHUNK_CHARS = 700  # target size of one passage


def split_paragraphs(content: str) -> list[str]:
    """Blank lines separate paragraphs. A very long one (a spreadsheet, a dense
    PDF page) is cut further at line breaks so no single passage is huge."""
    paragraphs = []
    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) <= CHUNK_CHARS:
            paragraphs.append(block)
            continue
        piece = ""
        for line in block.split("\n"):
            if piece and len(piece) + len(line) > CHUNK_CHARS:
                paragraphs.append(piece)
                piece = ""
            piece = f"{piece}\n{line}".strip()
        if piece:
            paragraphs.append(piece)
    return paragraphs


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Split every supported file in the folder into paragraph-sized passages."""
    chunks = []
    skipped = []
    for path in sorted(docs_dir.glob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue  # skip folders and Office lock files
        try:
            content = read_any(path)
        except Exception as error:
            print(f"  could not read {path.name}: {error}")
            continue
        if not content.strip():
            skipped.append(path.name)
            continue
        buffer = ""
        for paragraph in split_paragraphs(content):
            if buffer and len(buffer) + len(paragraph) > CHUNK_CHARS:
                chunks.append({"source": path.name, "text": buffer})
                buffer = ""
            buffer = f"{buffer}\n\n{paragraph}".strip()
        if buffer:
            chunks.append({"source": path.name, "text": buffer})
    if skipped:
        print(f"  no text found in: {', '.join(skipped)} (unsupported type, empty, or a scanned PDF)")
    return chunks


def gemini_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Turn text into vectors with the Gemini embedding model."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    vectors = []
    for start in range(0, len(texts), 50):  # small batches stay inside free-tier limits
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[start:start + 50],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in response.embeddings)
    return vectors


class Retriever:
    def __init__(self, docs_dir: Path = DOCS_DIR, index_file: Path = INDEX_FILE, embed=gemini_embed):
        self.embed = embed
        self.chunks = load_chunks(docs_dir)
        if not self.chunks:
            raise RuntimeError(f"No readable documents in {docs_dir}. Supported types: {SUPPORTED}")
        fingerprint = hashlib.sha256(
            json.dumps([EMBEDDING_MODEL, self.chunks], sort_keys=True).encode()
        ).hexdigest()

        cached = json.loads(index_file.read_text()) if index_file.exists() else {}
        if cached.get("fingerprint") == fingerprint:
            vectors = cached["vectors"]  # documents unchanged: reuse saved embeddings
        else:
            vectors = self.embed([c["text"] for c in self.chunks], "RETRIEVAL_DOCUMENT")
            index_file.write_text(json.dumps({"fingerprint": fingerprint, "vectors": vectors}))
        self.matrix = self._normalize(np.array(vectors, dtype=float))

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        return m / np.linalg.norm(m, axis=-1, keepdims=True)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        q = self._normalize(np.array(self.embed([query], "RETRIEVAL_QUERY")[0], dtype=float))
        scores = self.matrix @ q  # cosine similarity
        best = np.argsort(scores)[::-1][:top_k]
        return [
            {"source": self.chunks[i]["source"], "passage": self.chunks[i]["text"], "score": round(float(scores[i]), 3)}
            for i in best
        ]

    def sources(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})
