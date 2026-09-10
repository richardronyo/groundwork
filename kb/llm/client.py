"""
Unified LLM client supporting multiple providers via LiteLLM.
Handles API key storage & retrieval from the ai_providers table.
The model name is NOT stored – it is passed at call time.
"""

import os
import sys
import getpass
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from kb.relationaldb.initialize_db import get_connection

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MASTER_KEY = os.getenv("ENCRYPTION_KEY")
if not MASTER_KEY:
    print("Error: ENCRYPTION_KEY not set in .env or environment.")
    print("Generate one with: python3 -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
    print("Then add it to your .env as ENCRYPTION_KEY=<value>.")
    sys.exit(1)
cipher = Fernet(MASTER_KEY.encode())


class LLMCallError(RuntimeError):
    """Raised when an LLM completion fails, so callers can catch and retry
    instead of the process being killed outright."""
    pass


def get_stored_key(provider: str, user_id: int = 1) -> str | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT api_key_encrypted FROM ai_providers WHERE user_id=%s AND provider_name=%s",
                (user_id, provider)
            )
            row = cur.fetchone()
            if row:
                encrypted = row[0].encode()
                try:
                    return cipher.decrypt(encrypted).decode()
                except InvalidToken:
                    print(f"  ⚠️  Stored key for '{provider}' could not be decrypted "
                          f"(ENCRYPTION_KEY may have changed since it was saved) — "
                          f"you'll be re-prompted.")
                    return None
            return None
    finally:
        conn.close()


def store_api_key(provider: str, api_key: str, user_id: int = 1):
    encrypted = cipher.encrypt(api_key.encode()).decode()
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
    key = get_stored_key(provider, user_id)
    if key is None:
        key = prompt_for_key(provider)
    return key


def get_completion(
    messages: list[dict],
    model: str,
    provider: str = None,
    user_id: int = 1,
    **kwargs
) -> str:
    """
    Unified completion call using LiteLLM.
    provider is optional – if not given, we infer from model name (e.g., 'openai/gpt-4').
    kwargs are passed to litellm.completion (temperature, etc.).
    """
    import litellm

    # If provider is given, set the corresponding environment variable
    if provider:
        key = get_api_key(provider, user_id)
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
            "replicate": "REPLICATE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        env_var = env_map.get(provider.lower())
        if env_var:
            os.environ[env_var] = key
    else:
        # Infer provider from model prefix (e.g., "openai/gpt-4")
        if "/" in model:
            prov = model.split("/")[0]
            get_api_key(prov, user_id)

    # Now call LiteLLM
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    except Exception as e:
        # Raise (don't sys.exit) so call_llm()'s retry loop can catch this,
        # and so a single bad file/call doesn't kill the whole ingestion run.
        raise LLMCallError(f"LLM call failed ({provider or model}): {e}") from e

if __name__ == '__main__':
        # Example usage
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        model = "gpt-4o-mini"
        provider = "openai"
        store_api_key(provider, OPENAI_API_KEY)
        try:
            completion = get_completion(messages, model, provider)
            print("LLM Response:", completion)
        except LLMCallError as e:
            print(e)