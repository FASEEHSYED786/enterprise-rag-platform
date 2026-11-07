"""Document parsers: turn raw bytes into structured sections (heading path + text)."""
from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass
class Section:
    heading: str
    text: str


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "noscript"}
    HEAD = {"h1", "h2", "h3", "h4"}
    BLOCK = {"p", "li", "div", "br", "tr", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self._head, self._buf, self._skip, self._in_head = "", [], 0, False
        self._head_buf: list[str] = []

    def _flush(self):
        t = re.sub(r"[ \t]+", " ", "".join(self._buf)).strip()
        if t:
            self.sections.append(Section(self._head, t))
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.HEAD:
            self._flush()
            self._in_head, self._head_buf = True, []
        elif tag in self.BLOCK:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.HEAD:
            self._head, self._in_head = " ".join("".join(self._head_buf).split()), False

    def handle_data(self, data):
        if self._skip:
            return
        (self._head_buf if self._in_head else self._buf).append(data)

    def close(self):
        super().close()
        self._flush()


def parse_html(raw: str) -> list[Section]:
    p = _HTMLText()
    p.feed(raw)
    p.close()
    return p.sections


_MD_HEAD = re.compile(r"^(#{1,4})\s+(.*)$")


def parse_markdown(raw: str) -> list[Section]:
    sections, head, buf, stack = [], "", [], []
    in_code = False
    for line in raw.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _MD_HEAD.match(line)
        if m:
            if "".join(buf).strip():
                sections.append(Section(head, "\n".join(buf).strip()))
            level = len(m.group(1))
            stack = stack[: level - 1] + [m.group(2).strip()]
            head, buf = " > ".join(stack), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append(Section(head, "\n".join(buf).strip()))
    return sections


def parse_pdf(raw: bytes) -> list[Section]:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PDF support needs `pip install pypdf`") from e
    reader = PdfReader(io.BytesIO(raw))
    return [Section(f"page {i + 1}", t) for i, p in enumerate(reader.pages) if (t := (p.extract_text() or "").strip())]


def parse(filename: str, raw: bytes) -> list[Section]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(raw)
    text = raw.decode("utf-8", errors="replace")
    if name.endswith((".html", ".htm")):
        return parse_html(text)
    if name.endswith((".md", ".markdown")):
        return parse_markdown(text)
    return [Section("", html.unescape(text).strip())] if text.strip() else []
