"""Hybrid retrieval: BM25 (FTS5) + dense, fused with RRF, optional cross-encoder rerank, MMR diversification."""
from __future__ import annotations

from typing import Protocol

from ..store import Hit, Store
from .embeddings import Embedder, cosine


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    fused: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking):
            fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(h.chunk_id, h)
    out = []
    for cid in sorted(fused, key=lambda c: -fused[c]):
        h = by_id[cid]
        out.append(Hit(h.chunk_id, h.doc_name, h.heading, h.text, fused[cid], h.embedding))
    return out


def mmr(hits: list[Hit], vectors: dict[int, list[float]], k: int, lam: float = 0.7) -> list[Hit]:
    """Maximal Marginal Relevance: trade relevance against redundancy among selected chunks."""
    if len(hits) <= 1:
        return hits[:k]
    top = max(h.score for h in hits) or 1.0
    chosen: list[Hit] = []
    pool = list(hits)
    while pool and len(chosen) < k:
        def value(h: Hit) -> float:
            rel = h.score / top
            red = max((cosine(vectors[h.chunk_id], vectors[c.chunk_id]) for c in chosen), default=0.0) if h.chunk_id in vectors else 0.0
            return lam * rel - (1 - lam) * red

        best = max(pool, key=value)
        chosen.append(best)
        pool.remove(best)
    return chosen


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, reranker: Reranker | None = None):
        self.store, self.embedder, self.reranker = store, embedder, reranker

    def retrieve(self, tenant_id: str, query: str, k: int = 5, candidates: int = 25) -> list[Hit]:
        qvec = self.embedder.embed([query])[0]
        sparse = self.store.search_bm25(tenant_id, query, candidates)
        dense = self.store.search_dense(tenant_id, qvec, candidates)
        fused = rrf([sparse, dense])[: candidates]
        if self.reranker:
            fused = self.reranker.rerank(query, fused)
        vecs = self.store.embeddings_for(tenant_id, [h.chunk_id for h in fused])
        return mmr(fused, vecs, k)
