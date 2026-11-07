"""Shared text helpers."""
import re

WORD = re.compile(r"[a-z0-9]+")
_STOP_TEXT = (
    "the and for are was were what which who whom whose when where why how does did have has had can could would should "
    "with that this these those from into about than then there their them they you your our its not but any all per get "
    "got many much some more most is a an of to in on at by it be or as"
)
STOP = frozenset(_STOP_TEXT.split())


def content_words(text: str) -> set[str]:
    return {t for t in WORD.findall(text.lower()) if len(t) > 1 and t not in STOP}
