"""
Thin wrapper around an OpenAI-compatible client configured for OpenRouter.
- Caches calls by (model, system, prompt) hash to avoid redundant API calls.
- Returns (text, tokens_in, tokens_out, latency_ms).
- Used by all strategies via Strategy._timed_llm().
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set")

        _client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "agentic_lab",
            },
        )
    return _client


def _cache_key(model: str, system: str, prompt: str) -> str:
    raw = json.dumps({"model": model, "system": system, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def call_llm(
    model: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 1024,
    cache: Optional[dict] = None,
    stop_sequences: Optional[list[str]] = None,
) -> tuple[str, int, int, float]:
    key = _cache_key(model, system, prompt)
    if cache is not None and key in cache:
        cached = cache[key]
        return cached["text"], cached["tok_in"], cached["tok_out"], 0.0

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_sequences,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    text = response.choices[0].message.content or ""
    tok_in = response.usage.prompt_tokens if response.usage else 0
    tok_out = response.usage.completion_tokens if response.usage else 0

    if cache is not None:
        cache[key] = {"text": text, "tok_in": tok_in, "tok_out": tok_out}

    return text, tok_in, tok_out, latency_ms