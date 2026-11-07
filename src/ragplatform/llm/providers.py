"""LLM providers behind one interface: generate() and stream().

- ExtractiveLLM: offline, deterministic, no API key - picks the best-matching context sentences.
- AnthropicLLM: Anthropic Messages API (incl. SSE streaming) over httpx.
- OpenAICompatLLM: any OpenAI-compatible /chat/completions server (OpenAI, vLLM, Ollama, LiteLLM)."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Protocol

from ..config import Settings
from ..textutil import content_words


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...
    def stream(self, system: str, user: str) -> Iterator[str]: ...


class ExtractiveLLM:
    """Answers by extracting the most relevant context sentences and keeping their [n] markers."""

    def generate(self, system: str, user: str) -> str:
        question = user.rsplit("Question:", 1)[-1].strip()
        q = content_words(question)
        context = user.split("Context:", 1)[-1].rsplit("Question:", 1)[0]
        best: list[tuple[float, str]] = []
        for block in re.split(r"\n(?=\[\d+\])", context.strip()):
            m = re.match(r"\[(\d+)\]", block)
            if not m:
                continue
            body = block[m.end():].strip().split("\n", 1)[-1]  # drop the "(doc) heading" line
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                toks = content_words(sent)
                if toks and q:
                    best.append((len(q & toks) / len(q), f"{sent.strip()} [{m.group(1)}]"))
        best.sort(key=lambda x: -x[0])
        picked = [s for sc, s in best[:2] if sc > 0]
        return " ".join(picked) if picked else "I don't know based on the provided documents."

    def stream(self, system: str, user: str) -> Iterator[str]:
        for tok in self.generate(system, user).split(" "):
            yield tok + " "


class AnthropicLLM:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, max_tokens: int = 800):
        self.api_key, self.model, self.max_tokens = api_key, model, max_tokens

    def _body(self, system: str, user: str, stream: bool) -> dict:
        return {"model": self.model, "max_tokens": self.max_tokens, "system": system, "stream": stream,
                "messages": [{"role": "user", "content": user}]}

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def generate(self, system: str, user: str) -> str:
        import httpx

        r = httpx.post(self.URL, headers=self._headers(), json=self._body(system, user, False), timeout=60)
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    def stream(self, system: str, user: str) -> Iterator[str]:
        import httpx

        with httpx.stream("POST", self.URL, headers=self._headers(), json=self._body(system, user, True), timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data:"):
                    ev = json.loads(line[5:])
                    if ev.get("type") == "content_block_delta":
                        yield ev["delta"].get("text", "")


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    def _body(self, system: str, user: str, stream: bool) -> dict:
        return {"model": self.model, "stream": stream, "temperature": 0.1,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

    def generate(self, system: str, user: str) -> str:
        import httpx

        r = httpx.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                       json=self._body(system, user, False), timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def stream(self, system: str, user: str) -> Iterator[str]:
        import httpx

        with httpx.stream("POST", f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                          json=self._body(system, user, True), timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    delta = json.loads(line[5:])["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]


def build_llm(s: Settings) -> LLM:
    if s.llm_provider == "anthropic":
        return AnthropicLLM(s.llm_api_key, s.llm_model)
    if s.llm_provider == "openai":
        return OpenAICompatLLM(s.llm_base_url, s.llm_api_key, s.llm_model)
    return ExtractiveLLM()
