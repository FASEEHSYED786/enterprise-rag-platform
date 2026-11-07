"""Guardrails: prompt-injection screening of retrieved text, PII redaction, citation validation."""
from __future__ import annotations

import re

INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|prompts)|disregard (the )?(system|previous) prompt|"
    r"you are now |reveal (your|the) (system )?prompt|act as (an? )?(admin|developer|dan))",
    re.IGNORECASE,
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
CITE = re.compile(r"\[(\d+)\]")


def luhn(num: str) -> bool:
    ds = [int(c) for c in num if c.isdigit()][::-1]
    return len(ds) >= 13 and sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(ds)) % 10 == 0


def sanitize_context(text: str) -> tuple[str, bool]:
    """Neutralise instruction-like lines inside retrieved documents (indirect prompt injection)."""
    flagged = False
    lines = []
    for line in text.splitlines():
        if INJECTION.search(line):
            flagged = True
            lines.append("[removed: instruction-like content]")
        else:
            lines.append(line)
    return "\n".join(lines), flagged


def redact_pii(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = CARD.sub(lambda m: "[card]" if luhn(m.group()) else m.group(), text)
    return PHONE.sub("[phone]", text)


def valid_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Returns (valid, invalid) citation numbers; invalid ones point at sources that do not exist."""
    nums = sorted({int(m) for m in CITE.findall(answer)})
    return [n for n in nums if 1 <= n <= n_sources], [n for n in nums if not 1 <= n <= n_sources]
