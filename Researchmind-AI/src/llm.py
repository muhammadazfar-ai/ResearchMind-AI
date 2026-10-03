import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

# Each agent role has its own model preference list. Later entries are used if a model is
# unavailable or rate-limited. Override the first choice with GROQ_WRITER_MODEL / GROQ_FAST_MODEL.
DEFAULT_MODELS = {
    "writer": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"],
    "fast":   ["llama-3.1-8b-instant", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
}


class LLMError(Exception):
    """Raised when no Groq model could produce an answer."""


def model_candidates(role: str) -> list:
    models = list(DEFAULT_MODELS[role])
    override = os.getenv(f"GROQ_{role.upper()}_MODEL", "").strip()
    if override:
        models = [override] + [m for m in models if m != override]
    return models


def get_client():
    """Creates the Groq client lazily so a missing key is a clear runtime message, not an import crash."""
    key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise LLMError("`GROQ_API_KEY` is missing in your `.env` file (or Space secrets).")
    from groq import Groq

    return Groq(api_key=key)


def _is_rate_limit(error) -> bool:
    text = str(error).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def complete(messages: list, role: str = "fast", temperature: float = 0.0, json_mode: bool = False) -> str:
    """Single (non-streaming) completion with model fallback and one patient retry on rate limits."""
    client = get_client()
    last_error = None

    for attempt in range(2):
        for model_id in model_candidates(role):
            kwargs = {"model": model_id, "messages": messages, "temperature": temperature}
            try:
                if json_mode:
                    try:
                        response = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
                    except Exception as json_error:
                        if _is_rate_limit(json_error):
                            raise
                        response = client.chat.completions.create(**kwargs)
                else:
                    response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"[DEBUG NOTICE] Groq model '{model_id}' failed: {e}")
                last_error = e
        if attempt == 0 and last_error is not None and _is_rate_limit(last_error):
            time.sleep(8)
            continue
        break

    raise LLMError(f"All Groq models failed. Last error: `{last_error}`")


def stream_completion(messages: list, role: str = "writer", temperature: float = 0.3):
    """Yields text deltas, falling back to the next model if one fails before producing output."""
    client = get_client()
    last_error = None

    for attempt in range(2):
        for model_id in model_candidates(role):
            started = False
            try:
                stream = client.chat.completions.create(
                    model=model_id, messages=messages, temperature=temperature, stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        started = True
                        yield delta
                return
            except Exception as e:
                if started:
                    # Failed midway: switching models would restart the report, so surface the error.
                    raise LLMError(f"Generation was interrupted: `{e}`")
                print(f"[DEBUG NOTICE] Groq model '{model_id}' failed: {e}")
                last_error = e
        if attempt == 0 and last_error is not None and _is_rate_limit(last_error):
            time.sleep(8)
            continue
        break

    raise LLMError(f"All Groq models failed. Last error: `{last_error}`")
