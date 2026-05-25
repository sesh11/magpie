"""LLM dispatch for the MAGPIE exploration.

Stays independent of upstream `simulate_agents.py` so this module doesn't
drag in the top-level `import openai` (not present in this venv). The
OpenAI client is imported lazily inside the gpt5 branch only.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from google import genai
from google.genai import types


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    prompt_chars: int
    latency_ms: float
    model: str | None = None


def load_env_file() -> None:
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def get_gemini_api_keys() -> list[str]:
    keys: list[str] = []
    for i in range(1, 6):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    primary = os.getenv("GEMINI_API_KEY")
    if primary and primary not in keys:
        keys.append(primary)
    return keys


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _build_result(
    text: str,
    prompt: str,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    model: str | None,
    latency_ms: float,
) -> LLMResult:
    prompt_chars = len(prompt)
    resolved_input = input_tokens if input_tokens is not None else _estimate_tokens(prompt)
    resolved_output = output_tokens if output_tokens is not None else _estimate_tokens(text)
    resolved_total = (
        total_tokens if total_tokens is not None else resolved_input + resolved_output
    )
    return LLMResult(
        text=text or "",
        input_tokens=resolved_input,
        output_tokens=resolved_output,
        total_tokens=resolved_total,
        prompt_chars=prompt_chars,
        latency_ms=round(latency_ms, 1),
        model=model,
    )


load_env_file()
GEMINI_API_KEYS = get_gemini_api_keys()
model_name = os.getenv("GEMINI_MODEL_NAME")
if not model_name:
    raise RuntimeError("GEMINI_MODEL_NAME is not set")


def generate_gemini(
    prompt: str, model_name: str = model_name, temperature: float = 0.7
) -> LLMResult:
    if not model_name:
        raise RuntimeError("GEMINI_MODEL_NAME is not set")
    if not GEMINI_API_KEYS:
        raise RuntimeError(
            "No Gemini API keys found. Set GEMINI_API_KEY or GEMINI_API_KEY_1..5"
        )
    while True:
        try:
            selected = random.choice(GEMINI_API_KEYS)
            client = genai.Client(api_key=selected)
            started = time.perf_counter()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            latency_ms = (time.perf_counter() - started) * 1000
            usage = getattr(response, "usage_metadata", None)
            return _build_result(
                response.text or "",
                prompt,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                model=model_name,
                latency_ms=latency_ms,
            )
        except Exception:
            time.sleep(30)
            continue


def generate_gpt5(prompt: str, temperature: float = 0.7) -> LLMResult:
    # Lazy import — only fail if user actually picks gpt5.
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the gpt5 backend")
    client = OpenAI(api_key=api_key)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    usage = getattr(response, "usage", None)
    text = response.choices[0].message.content or ""
    return _build_result(
        text,
        prompt,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        model="gpt-5",
        latency_ms=latency_ms,
    )


def generate_anthropic(prompt: str, temperature: float = 0.7) -> LLMResult:
    # Lazy import — only fail if user actually picks anthropic. Env-var check
    # is also lazy so a gemini-only setup doesn't need ANTHROPIC_* configured.
    import anthropic  # type: ignore

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for the anthropic backend")
    anthropic_model = os.getenv("ANTHROPIC_MODEL_NAME")
    if not anthropic_model:
        raise RuntimeError("ANTHROPIC_MODEL_NAME is not set")

    # Opus 4.7 removes temperature/top_p/top_k entirely — sending them 400s.
    # Other Claude 4.x models still accept temperature.
    create_kwargs: dict = {
        "model": anthropic_model,
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not anthropic_model.startswith("claude-opus-4-7"):
        create_kwargs["temperature"] = temperature

    # max_retries=8: SDK does exponential backoff with jitter and respects the
    # retry-after header on 429s. Default is 2 — too low when running our prompts
    # (~10K input tokens each) against the 50K TPM org cap.
    # timeout=180: cumulative wait across retries can exceed the default 10-min
    # per-request budget, but we don't want a runaway hang either.
    client = anthropic.Anthropic(api_key=api_key, max_retries=8, timeout=180.0)

    # Outer fallback loop: if the SDK's 8 internal retries are still exhausted
    # under sustained TPM pressure, take a hard 60s nap (one full TPM window)
    # and try once more. Belt-and-suspenders for parallel-script runs.
    last_err: Exception | None = None
    for outer_attempt in range(3):
        try:
            started = time.perf_counter()
            response = client.messages.create(**create_kwargs)
            latency_ms = (time.perf_counter() - started) * 1000
            text = next((b.text for b in response.content if b.type == "text"), "")
            usage = getattr(response, "usage", None)
            return _build_result(
                text,
                prompt,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                total_tokens=None,
                model=anthropic_model,
                latency_ms=latency_ms,
            )
        except anthropic.RateLimitError as e:
            last_err = e
            sleep_s = 60 + random.uniform(0, 10)  # jitter to desync parallel callers
            print(
                f"⚠️  anthropic RateLimitError after SDK retries; "
                f"outer attempt {outer_attempt + 1}/3, sleeping {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def generate(prompt: str, llm_type: str, temperature: float = 0.7) -> LLMResult:
    if llm_type == "gemini":
        return generate_gemini(prompt, temperature=temperature)
    if llm_type == "gpt5":
        return generate_gpt5(prompt, temperature=temperature)
    if llm_type == "anthropic":
        return generate_anthropic(prompt, temperature=temperature)
    raise ValueError(f"Unknown llm_type: {llm_type}")
