"""Structure-aware chunking: split on paragraphs/sentences to a token budget, with overlap,
and prefix each chunk with its heading path so retrieval keeps context."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .parsers import Section

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def count_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class Chunk:
    ordinal: int
    heading: str
    text: str
    content_hash: str


def _units(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        out.extend(_SENT.split(para) if count_tokens(para) > 120 else [para])
    return out


def _make_chunk(ordinal: int, heading: str, parts: list[str]) -> Chunk | None:
    body = " ".join(parts).strip()
    if not body:
        return None
    text = f"{heading}\n{body}" if heading else body
    return Chunk(ordinal, heading, text, hashlib.sha256(text.encode()).hexdigest()[:16])


def chunk_sections(sections: list[Section], max_tokens: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        cur: list[str] = []
        cur_tok = 0
        for u in _units(sec.text):
            t = count_tokens(u)
            if cur and cur_tok + t > max_tokens:
                if c := _make_chunk(len(chunks), sec.heading, cur):
                    chunks.append(c)
                tail: list[str] = []
                tail_tok = 0
                for prev in reversed(cur):  # carry a token-budgeted tail into the next chunk
                    if tail_tok + count_tokens(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_tok += count_tokens(prev)
                cur, cur_tok = tail, tail_tok
            cur.append(u)
            cur_tok += t
        if c := _make_chunk(len(chunks), sec.heading, cur):
            chunks.append(c)
    return chunks
