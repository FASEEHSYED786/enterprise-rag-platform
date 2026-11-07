"""Offline regression gate: ingests a fixture corpus, runs golden questions and fails CI
if retrieval recall, answer grounding or refusal behaviour regress below thresholds.

    python evals/run_eval.py            # prints a report, exit code 1 on regression
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragplatform.config import Settings
from ragplatform.llm.providers import ExtractiveLLM
from ragplatform.pipeline import RagPipeline
from ragplatform.retrieval.embeddings import HashEmbedder
from ragplatform.store import Store

HERE = pathlib.Path(__file__).parent
THRESHOLDS = {"recall_at_k": 0.9, "grounded_rate": 0.9, "refusal_rate": 1.0}


def main() -> int:
    s = Settings(db_path=":memory:")
    store = Store(":memory:")
    store.create_tenant("eval", "Eval")
    pipe = RagPipeline(store, HashEmbedder(s.embed_dim), ExtractiveLLM(), s)
    for f in sorted((HERE / "corpus").glob("*.md")):
        pipe.ingest("eval", f.name, f.read_bytes())

    cases = [json.loads(line) for line in (HERE / "golden.jsonl").read_text().splitlines() if line.strip()]
    answerable = [c for c in cases if c["expected_doc"]]
    unanswerable = [c for c in cases if not c["expected_doc"]]

    recall = grounded = 0
    for c in answerable:
        ans = pipe.ask("eval", c["question"], k=3)
        recall += any(src["document"] == c["expected_doc"] for src in ans.sources)
        grounded += ans.grounded and all(t.lower() in ans.answer.lower() for t in c["must_contain"])
    refused = sum(not pipe.ask("eval", c["question"]).grounded for c in unanswerable)

    result = {
        "recall_at_k": recall / len(answerable),
        "grounded_rate": grounded / len(answerable),
        "refusal_rate": refused / max(1, len(unanswerable)),
        "cases": len(cases),
    }
    print(json.dumps(result, indent=2))
    failed = {k: v for k, v in result.items() if k in THRESHOLDS and v < THRESHOLDS[k]}
    if failed:
        print(f"REGRESSION: {failed} below thresholds {THRESHOLDS}")
        return 1
    print("eval gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
