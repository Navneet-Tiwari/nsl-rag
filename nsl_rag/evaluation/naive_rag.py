"""
naive_rag.py
------------
NaiveRAG — baseline RAG implementation for benchmark comparison.
Uses chunking + cosine similarity retrieval. No reasoning layer.
No lattice. No symbolic constraints.

This is what NSL-RAG is compared against in benchmarks.

Usage:
    from nsl_rag.evaluation.naive_rag import NaiveRAG
    naive = NaiveRAG()
    result = naive.query("Why is the payment service failing?")
"""

import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from nsl_rag.config.config_loader import config
from nsl_rag.core.logger import get_logger
from nsl_rag.data.ecommerce import EcommerceSystem

log = get_logger(__name__)


# ── Naive RAG Result ──────────────────────────────────────────────────────────


@dataclass
class NaiveRAGResult:
    """Result from naive RAG query."""

    answer: str
    retrieved_chunks: list[str]
    tokens_used: int
    latency_ms: float
    query: str
    chunk_count: int = 0
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "chunks_retrieved": self.chunk_count,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "query": self.query,
        }


# ── Document Chunker ──────────────────────────────────────────────────────────


class DocumentChunker:
    """
    Splits raw documents into fixed-size chunks.
    Simulates standard RAG ingestion pipeline.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[str]) -> list[str]:
        """Split documents into overlapping chunks."""
        chunks = []
        for doc in documents:
            start = 0
            while start < len(doc):
                end = start + self.chunk_size
                chunks.append(doc[start:end])
                start += self.chunk_size - self.chunk_overlap
        log.debug("Chunked %d documents → %d chunks", len(documents), len(chunks))
        return chunks


# ── Simple Embedder ───────────────────────────────────────────────────────────


class SimpleEmbedder:
    """
    Lightweight TF-IDF style embedder.
    Uses numpy only — no external models needed.
    Sufficient for baseline comparison demo.
    """

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._idf: np.ndarray = np.array([])
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary from corpus."""
        vocab_set = set()
        for text in texts:
            for word in text.lower().split():
                word = word.strip(".,!?;:()[]")
                if len(word) > 2:
                    vocab_set.add(word)

        self._vocab = sorted(list(vocab_set))

        # Compute IDF
        n_docs = len(texts)
        idf_scores = []
        for term in self._vocab:
            doc_count = sum(1 for t in texts if term in t.lower())
            idf = np.log((n_docs + 1) / (doc_count + 1)) + 1
            idf_scores.append(idf)

        self._idf = np.array(idf_scores)
        self._fitted = True
        log.debug("Embedder fitted — vocab size: %d", len(self._vocab))

    def embed(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")

        words = [w.lower().strip(".,!?;:()[]") for w in text.split()]
        word_count = len(words) if words else 1

        tf = np.zeros(len(self._vocab))
        for i, term in enumerate(self._vocab):
            count = words.count(term)
            tf[i] = count / word_count

        tfidf = tf * self._idf

        # Normalize
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm

        return tfidf

    def cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


# ── Naive RAG ─────────────────────────────────────────────────────────────────


class NaiveRAG:
    """
    Baseline RAG implementation.

    Pipeline:
        1. Chunk all documents
        2. Embed chunks with TF-IDF
        3. Embed query
        4. Find top-k similar chunks by cosine similarity
        5. Concatenate chunks as context
        6. Simulate LLM generation (token counting)

    No reasoning. No validation. No lattice.
    Pure similarity search — the standard RAG baseline.
    """

    def __init__(self) -> None:
        self._chunk_size = config.evaluation.naive_rag.chunk_size
        self._chunk_overlap = config.evaluation.naive_rag.chunk_overlap
        self._top_k = config.evaluation.naive_rag.top_k
        self._chunker = DocumentChunker(self._chunk_size, self._chunk_overlap)
        self._embedder = SimpleEmbedder()
        self._chunks: list[str] = []
        self._chunk_vectors: list[np.ndarray] = []
        self._built = False
        log.debug(
            "NaiveRAG initialised — chunk_size: %d, top_k: %d",
            self._chunk_size,
            self._top_k,
        )

    def build_index(self) -> None:
        """
        Build the naive RAG index from e-commerce data.
        Converts all node content into flat text documents.
        """
        log.info("Building NaiveRAG index...")
        start = time.time()

        raw_nodes = EcommerceSystem.get_raw_nodes()
        documents = [
            f"{node['title']}: {node['summary']} {node['content']}"
            for node in raw_nodes
        ]

        self._chunks = self._chunker.chunk(documents)
        self._embedder.fit(self._chunks)
        self._chunk_vectors = [self._embedder.embed(chunk) for chunk in self._chunks]

        self._built = True
        elapsed = (time.time() - start) * 1000

        log.info(
            "NaiveRAG index built — %d chunks in %.1fms", len(self._chunks), elapsed
        )

    def query(self, query: str) -> NaiveRAGResult:
        """
        Run naive RAG retrieval for a query.
        Returns result with token count and latency for benchmarking.
        """
        if not self._built:
            self.build_index()

        start = time.time()

        # Embed query
        query_vec = self._embedder.embed(query)

        # Find top-k chunks by cosine similarity
        scores = [
            self._embedder.cosine_similarity(query_vec, chunk_vec)
            for chunk_vec in self._chunk_vectors
        ]

        top_indices = np.argsort(scores)[::-1][: self._top_k]
        retrieved = [self._chunks[i] for i in top_indices]

        # Simulate context assembly
        context = "\n".join(retrieved)
        tokens_used = self._estimate_tokens(query + context)
        latency_ms = (time.time() - start) * 1000

        # Simulate answer — in real system this would call LLM
        # For benchmarking we measure retrieval quality and token cost
        simulated_answer = (
            f"Based on {len(retrieved)} retrieved chunks: "
            f"{retrieved[0][:200] if retrieved else 'No relevant content found'}"
        )

        log.debug(
            "NaiveRAG query complete — %d chunks, %d tokens, %.1fms",
            len(retrieved),
            tokens_used,
            latency_ms,
        )

        return NaiveRAGResult(
            answer=simulated_answer,
            retrieved_chunks=retrieved,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            query=query,
            chunk_count=len(retrieved),
        )

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        Approximation: 1 token ≈ 4 characters.
        """
        return len(text) // 4
