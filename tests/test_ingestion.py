from ragplatform.ingestion.chunking import chunk_sections, count_tokens
from ragplatform.ingestion.parsers import parse, parse_html, parse_markdown


def test_markdown_heading_paths():
    secs = parse_markdown("# A\nintro\n## B\nbody text\n```\n# not a heading\n```\n## C\nmore")
    assert [s.heading for s in secs] == ["A", "A > B", "A > C"]
    assert "# not a heading" in secs[1].text


def test_html_strips_script_and_nav():
    html = "<nav>menu</nav><h1>Title</h1><p>Hello <b>world</b></p><script>evil()</script>"
    secs = parse_html(html)
    assert secs[0].heading == "Title" and "evil" not in secs[0].text and "menu" not in secs[0].text


def test_parse_dispatch_and_empty():
    assert parse("x.txt", b"   ") == []
    assert parse("x.md", b"# H\ntext")[0].heading == "H"


def test_chunking_respects_budget_overlap_and_headings():
    para = " ".join(f"Sentence number {i} talks about topic {i % 5}." for i in range(80))
    chunks = chunk_sections(parse_markdown(f"# Doc\n{para}"), max_tokens=60, overlap=15)
    assert len(chunks) > 3
    assert all(c.text.startswith("Doc\n") for c in chunks)
    assert all(count_tokens(c.text) <= 60 + 15 for c in chunks)
    assert chunks[0].text != chunks[1].text and [c.ordinal for c in chunks] == list(range(len(chunks)))
