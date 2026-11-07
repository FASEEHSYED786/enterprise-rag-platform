"""Embedding backends. HashEmbedder is deterministic/offline (tests, demos);
HTTPEmbedder talks to any OpenAI-compatible /embeddings endpoint (OpenAI, vLLM, Ollama, TEI)."""
from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Protocol

TOK = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = TOK.findall(text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]  # unigrams + bigrams
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class HTTPEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int):
        self.base_url, self.api_key, self.model, self.dim = base_url.rstrip("/"), api_key, model, dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        r = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        r.raise_for_status()
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
