"""Tenant-isolated storage on SQLite (FTS5 for BM25 + embedding blobs for dense search).

Every read and write takes a tenant_id; there is no code path that queries across tenants.
For production scale, swap this class for the Postgres/pgvector implementation described in
docs/architecture.md - the interface (add_document / search_* / delete_document) is the seam."""
from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass

from .ingestion.chunking import Chunk
from .retrieval.embeddings import pack, unpack

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS api_keys (key_hash TEXT PRIMARY KEY, tenant_id TEXT NOT NULL REFERENCES tenants(id), role TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, name TEXT NOT NULL, content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL, created REAL NOT NULL, UNIQUE (tenant_id, name));
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL, heading TEXT NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tenant_id UNINDEXED, tokenize='porter unicode61');
"""

_WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass
class Hit:
    chunk_id: int
    doc_name: str
    heading: str
    text: str
    score: float
    embedding: list[float] | None = None


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)

    # --- tenants & keys -------------------------------------------------
    def create_tenant(self, tenant_id: str, name: str) -> str:
        """Creates a tenant and returns a fresh user API key (shown once, stored hashed)."""
        key = f"rk_{tenant_id}_{secrets.token_urlsafe(24)}"
        with self.lock, self.db:
            self.db.execute("INSERT INTO tenants VALUES (?,?,?)", (tenant_id, name, time.time()))
            self.db.execute("INSERT INTO api_keys VALUES (?,?,?)", (_hash_key(key), tenant_id, "user"))
        return key

    def authenticate(self, key: str) -> tuple[str, str] | None:
        with self.lock:
            row = self.db.execute("SELECT tenant_id, role FROM api_keys WHERE key_hash=?", (_hash_key(key),)).fetchone()
        return (row["tenant_id"], row["role"]) if row else None

    # --- documents ------------------------------------------------------
    def add_document(self, tenant_id: str, name: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
        content_hash = hashlib.sha256("".join(c.content_hash for c in chunks).encode()).hexdigest()[:16]
        with self.lock, self.db:
            old = self.db.execute("SELECT id, content_hash FROM documents WHERE tenant_id=? AND name=?", (tenant_id, name)).fetchone()
            if old and old["content_hash"] == content_hash:
                return {"id": old["id"], "name": name, "chunks": len(chunks), "status": "unchanged"}
            if old:  # re-ingest replaces the previous version atomically
                self._delete_doc(tenant_id, old["id"])
            cur = self.db.execute(
                "INSERT INTO documents(tenant_id,name,content_hash,chunk_count,created) VALUES (?,?,?,?,?)",
                (tenant_id, name, content_hash, len(chunks), time.time()),
            )
            doc_id = cur.lastrowid
            for c, v in zip(chunks, vectors):
                cid = self.db.execute(
                    "INSERT INTO chunks(tenant_id,doc_id,ordinal,heading,text,embedding) VALUES (?,?,?,?,?,?)",
                    (tenant_id, doc_id, c.ordinal, c.heading, c.text, pack(v)),
                ).lastrowid
                self.db.execute("INSERT INTO chunks_fts(rowid,text,tenant_id) VALUES (?,?,?)", (cid, c.text, tenant_id))
        return {"id": doc_id, "name": name, "chunks": len(chunks), "status": "replaced" if old else "created"}

    def _delete_doc(self, tenant_id: str, doc_id: int) -> None:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=? AND tenant_id=?", (doc_id, tenant_id))]
        for cid in ids:
            self.db.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
        self.db.execute("DELETE FROM chunks WHERE doc_id=? AND tenant_id=?", (doc_id, tenant_id))
        self.db.execute("DELETE FROM documents WHERE id=? AND tenant_id=?", (doc_id, tenant_id))

    def delete_document(self, tenant_id: str, doc_id: int) -> bool:
        with self.lock, self.db:
            exists = self.db.execute("SELECT 1 FROM documents WHERE id=? AND tenant_id=?", (doc_id, tenant_id)).fetchone()
            if exists:
                self._delete_doc(tenant_id, doc_id)
            return bool(exists)

    def list_documents(self, tenant_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT id,name,chunk_count,created FROM documents WHERE tenant_id=? ORDER BY id", (tenant_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- retrieval primitives ---------------------------------------------
    def search_bm25(self, tenant_id: str, query: str, limit: int = 20) -> list[Hit]:
        terms = _WORD.findall(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        with self.lock:
            rows = self.db.execute(
                """SELECT c.id, d.name, c.heading, c.text, bm25(chunks_fts) AS s
                   FROM chunks_fts f JOIN chunks c ON c.id = f.rowid JOIN documents d ON d.id = c.doc_id
                   WHERE chunks_fts MATCH ? AND f.tenant_id = ? AND c.tenant_id = ?
                   ORDER BY s LIMIT ?""",
                (match, tenant_id, tenant_id, limit),
            ).fetchall()
        return [Hit(r["id"], r["name"], r["heading"], r["text"], -r["s"]) for r in rows]

    def search_dense(self, tenant_id: str, qvec: list[float], limit: int = 20) -> list[Hit]:
        with self.lock:
            rows = self.db.execute(
                "SELECT c.id, d.name, c.heading, c.text, c.embedding FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.tenant_id=?",
                (tenant_id,),
            ).fetchall()
        scored = []
        for r in rows:
            v = unpack(r["embedding"])
            scored.append(Hit(r["id"], r["name"], r["heading"], r["text"], sum(a * b for a, b in zip(qvec, v)), v))
        scored.sort(key=lambda h: -h.score)
        return scored[:limit]

    def embeddings_for(self, tenant_id: str, ids: list[int]) -> dict[int, list[float]]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        with self.lock:
            rows = self.db.execute(f"SELECT id, embedding FROM chunks WHERE tenant_id=? AND id IN ({q})", (tenant_id, *ids)).fetchall()
        return {r["id"]: unpack(r["embedding"]) for r in rows}
