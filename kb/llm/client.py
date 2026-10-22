"""
Groundwork — LLM layer on LangChain's init_chat_model.

What this module owns:
  - API key storage in the `ai_providers` table, Fernet-encrypted (unchanged
    behaviour from the LiteLLM version, so existing stored keys keep working).
  - get_chat_model(): one LangChain chat model for any provider init_chat_model
    supports, with the stored key injected, optional rate limiting, and retries.
  - structured(): the same model bound to a Pydantic schema via
    .with_structured_output(), which replaces hand-parsing JSON out of replies.
  - get_completion(): the old string-in/string-out function, kept so any caller
    not yet moved to LangChain messages still works.

Model name is NOT stored — it is passed at call time, same as before.

Provider packages: init_chat_model needs the integration package for the
provider you use (langchain-openai for "openai", langchain-anthropic for
"anthropic", ...). LiteLLM is no longer used.
"""

from __future__ import annotations

import os
import sys
import getpass
from functools import lru_cache
from typing import TypeVar

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from pydantic import BaseModel

from kb.relationaldb.initialize_db import get_connection

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROVIDER = os.getenv("GROUNDWORK_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("GROUNDWORK_MODEL", "gpt-4o")

# Env vars each provider's LangChain integration reads. Used only as a
# fallback source for keys (stored key wins), and for the __main__ demo.
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
}

# Providers that need no API key (local servers).
KEYLESS_PROVIDERS = {"ollama"}


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails after retries, so callers can skip one
    file instead of the whole run dying."""


# ── Encrypted key store ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _cipher() -> Fernet:
    """Built on first use rather than at import time, so modules that import
    this one (tests, the TUI, read-only tools) don't need ENCRYPTION_KEY
    unless they actually touch stored keys."""
    master = os.getenv("ENCRYPTION_KEY")
    if not master:
        print("Error: ENCRYPTION_KEY not set in .env or environment.")
        print("Generate one with: python3 -c \"from cryptography.fernet import Fernet; "
              "print(Fernet.generate_key().decode())\"")
        print("Then add it to your .env as ENCRYPTION_KEY=<value>.")
        sys.exit(1)
    return Fernet(master.encode())


def get_stored_key(provider: str, user_id: int = 1) -> str | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT api_key_encrypted FROM ai_providers WHERE user_id=%s AND provider_name=%s",
                (user_id, provider),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        return _cipher().decrypt(row[0].encode()).decode()
    except InvalidToken:
        print(f"  ⚠️  Stored key for '{provider}' could not be decrypted "
              f"(ENCRYPTION_KEY may have changed since it was saved) — you'll be re-prompted.")
        return None


def store_api_key(provider: str, api_key: str, user_id: int = 1):
    encrypted = _cipher().encrypt(api_key.encode()).decode()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ai_providers (user_id, provider_name, api_key_encrypted)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, provider_name) DO UPDATE
                SET api_key_encrypted = EXCLUDED.api_key_encrypted,
                    updated_at = NOW()
            """, (user_id, provider, encrypted))
        conn.commit()
    finally:
        conn.close()


def prompt_for_key(provider: str) -> str:
    print(f"\n🔑 No API key found for provider '{provider}'.")
    key = getpass.getpass(f"Enter your {provider} API key: ").strip()
    if not key:
        print("Key cannot be empty.")
        sys.exit(1)
    store_api_key(provider, key)
    return key


def get_api_key(provider: str, user_id: int = 1) -> str:
    """Stored key → provider env var → interactive prompt (which stores it)."""
    key = get_stored_key(provider, user_id)
    if key is None:
        key = os.getenv(PROVIDER_ENV.get(provider.lower(), ""), "") or None
    if key is None:
        key = prompt_for_key(provider)
    return key


# ── Chat models ───────────────────────────────────────────────────────────────

def _split_provider(model: str, provider: str | None) -> tuple[str, str]:
    """Accepts both ("gpt-4o", "openai") and the LiteLLM-style "openai/gpt-4o"
    or init_chat_model-style "openai:gpt-4o", so old CLI invocations keep working."""
    if provider:
        return provider.lower(), model
    for sep in (":", "/"):
        if sep in model:
            prov, name = model.split(sep, 1)
            return prov.lower(), name
    return DEFAULT_PROVIDER, model


def make_rate_limiter(calls_per_minute: int):
    """One limiter shared by every model built with it — pass the same object
    to every get_chat_model() call in a run so the limit is global, the way
    the old RateLimiter class was shared across worker threads."""
    if not calls_per_minute or calls_per_minute <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter
    return InMemoryRateLimiter(
        requests_per_second=calls_per_minute / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=max(1, calls_per_minute // 10),
    )


def get_chat_model(model: str = DEFAULT_MODEL, provider: str | None = None,
                   temperature: float | None = None, user_id: int = 1,
                   rate_limiter=None, max_retries: int = 3, **kwargs):
    """
    A LangChain chat model for any provider init_chat_model supports, with the
    stored API key injected directly (no os.environ mutation, so two providers
    can be used in one process).

    max_retries is passed to the provider client, which retries rate-limit and
    transient errors with backoff — that replaces call_llm()'s manual loop.
    """
    from langchain.chat_models import init_chat_model

    prov, name = _split_provider(model, provider)
    params = dict(kwargs)
    if prov not in KEYLESS_PROVIDERS:
        params["api_key"] = get_api_key(prov, user_id)
    if temperature is not None:
        params["temperature"] = temperature
    if rate_limiter is not None:
        params["rate_limiter"] = rate_limiter
    params.setdefault("max_retries", max_retries)
    return init_chat_model(name, model_provider=prov, **params)


def structured(llm, schema: type[T], retries: int = 2):
    """
    Binds a chat model to a Pydantic schema. The returned runnable's
    .invoke(messages) returns a validated `schema` instance.

    LLM-facing schemas in this project avoid default values and free-form
    dicts, because OpenAI's strict json_schema mode rejects both.

    `retries` re-asks when the model's output fails validation (on top of the
    transport-level retries already inside the client).
    """
    runnable = llm.with_structured_output(schema)
    return runnable.with_retry(stop_after_attempt=retries + 1) if retries else runnable


def invoke_structured(llm, schema: type[T], messages, retries: int = 2) -> T:
    try:
        return structured(llm, schema, retries).invoke(messages)
    except Exception as e:
        raise LLMCallError(f"Structured LLM call failed ({schema.__name__}): {e}") from e


def invoke_text(llm, messages) -> str:
    try:
        return _content_text(llm.invoke(messages).content)
    except Exception as e:
        raise LLMCallError(f"LLM call failed: {e}") from e


def _content_text(content) -> str:
    """AIMessage.content is a str for most providers, but a list of content
    blocks for some (Anthropic with tools, reasoning models)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


# ── Graph plumbing ────────────────────────────────────────────────────────────

def llm_from_config(config: dict | None, temperature: float | None = None):
    """
    How every LangGraph node in Groundwork gets its model. Graphs are invoked
    with a RunnableConfig whose "configurable" dict holds either:
      - "llm": a ready chat model (tests inject a fake one here), or
      - "model" / "provider" / "rate_limiter": built on demand and cached on
        the config so one run reuses one client per temperature.
    Keeping the model out of graph state keeps state JSON-serialisable for
    checkpointers.
    """
    cfg = (config or {}).get("configurable", {})
    if cfg.get("llm") is not None:
        return cfg["llm"]
    cache = cfg.setdefault("_llm_cache", {})
    key = temperature
    if key not in cache:
        cache[key] = get_chat_model(
            cfg.get("model", DEFAULT_MODEL), cfg.get("provider"),
            temperature=temperature, rate_limiter=cfg.get("rate_limiter"))
    return cache[key]


def run_config(model: str = DEFAULT_MODEL, provider: str | None = None,
               rate_limit: int = 0, max_concurrency: int | None = None,
               thread_id: str | None = None, llm=None, **extra) -> dict:
    """Builds the RunnableConfig every Groundwork graph expects."""
    configurable = {"model": model, "provider": provider,
                    "rate_limiter": make_rate_limiter(rate_limit), **extra}
    if llm is not None:
        configurable["llm"] = llm
    if thread_id:
        configurable["thread_id"] = thread_id
    config = {"configurable": configurable, "recursion_limit": 200}
    if max_concurrency:
        config["max_concurrency"] = max_concurrency
    return config


# ── Backward-compatible string API ────────────────────────────────────────────

def get_completion(messages: list[dict], model: str, provider: str = None,
                   user_id: int = 1, **kwargs) -> str:
    """Old LiteLLM-era signature: OpenAI-style message dicts in, text out.
    kwargs (temperature, max_tokens, ...) go to the chat model."""
    llm = get_chat_model(model, provider, user_id=user_id, **kwargs)
    return invoke_text(llm, messages)


if __name__ == "__main__":
    provider, model = "openai", "gpt-4o-mini"
    env_key = os.getenv(PROVIDER_ENV[provider])
    if env_key:
        store_api_key(provider, env_key)
    llm = get_chat_model(model, provider)
    print("LLM Response:", invoke_text(llm, [("system", "You are a helpful assistant."),
                                             ("user", "Hello, how are you?")]))
