"""
Medical knowledge retriever for MedQA-USMLE.

Uses pre-built NumpyVectorStore (384-dim all-MiniLM-L6-v2) for fast
cosine-similarity retrieval over 76K USMLE textbook chunks.

No Qdrant needed — flat numpy arrays, loaded once, shared across all questions.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = REPO_ROOT / "rag_research" / "index"


class MedicalRetriever:
    """Singleton retriever — loads index once, reuses across questions."""

    _instance: Optional["MedicalRetriever"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, index_dir: str | Path = INDEX_DIR, device: str = "cuda"):
        if self._initialized:
            return
        self.index_dir = Path(index_dir)
        self.device = device
        self._embeddings: Optional[np.ndarray] = None
        self._chunks: list[dict] = []
        self._embedder = None
        self._initialized = True

    def _ensure_loaded(self):
        """Lazy load: embeddings + chunk metadata."""
        if self._embeddings is not None:
            return

        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "chunks_meta.jsonl"

        if not emb_path.exists():
            raise FileNotFoundError(f"Index not found at {emb_path}")

        import json as _json

        # Load chunks first (for metadata)
        print(f"[Retriever] Loading chunk metadata from {meta_path}...")
        t0 = time.time()
        self._chunks = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self._chunks.append(_json.loads(line))
        print(f"  {len(self._chunks)} chunks ({time.time() - t0:.1f}s)")

        # Load embeddings
        print(f"[Retriever] Loading embeddings from {emb_path}...")
        t0 = time.time()
        self._embeddings = np.load(str(emb_path))
        print(f"  Shape {self._embeddings.shape} ({time.time() - t0:.1f}s)")

    def _ensure_embedder(self):
        """Lazy load the SentenceTransformer for query encoding."""
        if self._embedder is not None:
            return
        print(f"[Retriever] Loading SentenceTransformer (all-MiniLM-L6-v2, {self.device})...")
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=self.device,
        )
        print(f"  Loaded in {time.time() - t0:.1f}s")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[list[str]] = None,
        include_options: Optional[dict[str, str]] = None,
    ) -> list[dict]:
        """Retrieve top-k medical chunks for a query.

        Args:
            query: Question stem or search query
            k: Number of results
            threshold: Minimum similarity (0.0 = no filter)
            source_filter: Optional list of source names
            include_options: If provided, append options to query for better retrieval

        Returns:
            List of dicts with keys: chunk_id, source, text, score
        """
        self._ensure_loaded()
        self._ensure_embedder()

        # Build query with options for better context
        full_query = query
        if include_options:
            opt_text = "\n".join(
                f"{k}. {v}" for k, v in sorted(include_options.items())
            )
            full_query = f"{query}\n{opt_text}"

        # Embed query
        t0 = time.time()
        query_emb = self._embedder.encode(
            full_query, convert_to_numpy=True, normalize_embeddings=True
        )
        embed_time = time.time() - t0

        # Cosine similarity search
        scores = np.dot(self._embeddings, query_emb)
        scores = np.clip(scores, -1.0, 1.0)
        top_indices = np.argsort(scores)[::-1]

        results = []
        seen_texts = set()
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue
            chunk = self._chunks[idx]
            source = chunk.get("source", "")
            if source_filter and source not in source_filter:
                continue
            # Dedup near-identical chunks
            text_preview = chunk["text"][:100]
            if text_preview in seen_texts:
                continue
            seen_texts.add(text_preview)

            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "source": source,
                "text": chunk["text"],
                "score": round(score, 4),
            })
            if len(results) >= k:
                break

        return results

    def format_context(self, chunks: list[dict], max_chars: int = 4000) -> str:
        """Format retrieved chunks into a context string for LLM prompt."""
        parts = []
        total = 0
        for i, c in enumerate(chunks):
            header = f"[Source {i+1}: {c['source']}]"
            text = c["text"]
            entry = f"{header}\n{text}"
            if total + len(entry) > max_chars:
                remaining = max_chars - total - len(header) - 10
                if remaining > 100:
                    parts.append(f"{header}\n{text[:remaining]}...")
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)

    @property
    def num_chunks(self) -> int:
        self._ensure_loaded()
        return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        self._ensure_loaded()
        seen = set()
        return sorted(set(
            c.get("source", "") for c in self._chunks if c.get("source")
        ))
