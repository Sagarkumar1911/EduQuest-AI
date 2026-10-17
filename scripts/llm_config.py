import os
import re
import time

from groq import APIConnectionError, APIStatusError, AuthenticationError


GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b").strip()
GROQ_TEXT_FALLBACK_MODEL = os.getenv(
    "GROQ_TEXT_FALLBACK_MODEL",
    "qwen/qwen3.6-27b",
).strip()
GROQ_VISION_MODELS = os.getenv(
    "GROQ_VISION_MODELS",
    "qwen/qwen3.6-27b,qwen/qwen3.8-27b",
)


def _configured_models():
    return [GROQ_TEXT_MODEL, GROQ_TEXT_FALLBACK_MODEL]


def vision_models():
    return [model.strip() for model in GROQ_VISION_MODELS.split(",") if model.strip()]


def _error_text(error):
    parts = [str(error)]
    code = getattr(error, "code", None)
    if code:
        parts.append(str(code))
    body = getattr(error, "body", None)
    if body is not None:
        parts.append(str(body))
    return " ".join(parts).casefold()


def _is_model_unavailable(error):
    text = _error_text(error)
    is_model_404 = (
        getattr(error, "status_code", None) == 404
        and "model" in text
        and ("not found" in text or "does not exist" in text)
    )
    return (
        "model_not_found" in text
        or "model_decommissioned" in text
        or is_model_404
    )


def _is_response_format_rejection(error):
    if getattr(error, "status_code", None) != 400:
        return False
    text = _error_text(error)
    return "response_format" in text or "json_object" in text or "json mode" in text


def _retry_delay(error):
    if isinstance(error, AuthenticationError):
        return None
    if isinstance(error, APIConnectionError):
        return 2
    if isinstance(error, APIStatusError):
        status_code = error.status_code
        if status_code == 429:
            return 2
        if 500 <= status_code <= 599:
            return 2
    return None


def _clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def chat_with_fallback(client, messages, **kwargs) -> str:
    models = list(dict.fromkeys(model for model in _configured_models() if model))
    if not models:
        raise RuntimeError("No Groq text models are configured.")

    kwargs.setdefault("reasoning_format", "hidden")
    models_tried = []
    last_error = None

    for model in models:
        models_tried.append(model)
        for attempt, wait_seconds in enumerate((None, 2, 5)):
            if wait_seconds is not None:
                time.sleep(wait_seconds)
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    **kwargs,
                )
                content = response.choices[0].message.content
                print(f"Groq answered with model: {model}")
                return _clean_response(content)
            except Exception as error:
                last_error = error
                if _is_model_unavailable(error) or _is_response_format_rejection(error):
                    break
                if isinstance(error, AuthenticationError):
                    raise RuntimeError(
                        f"Groq authentication failed for model {model}: {error}"
                    ) from error

                delay = _retry_delay(error)
                if delay is not None and attempt < 2:
                    continue
                if delay is not None:
                    break
                raise RuntimeError(
                    f"Groq request failed for model {model}: {error}"
                ) from error

    raise RuntimeError(
        f"All configured Groq models failed. Models tried: {', '.join(models_tried)}. "
        f"Last error: {last_error}"
    ) from last_error


def check_models(client):
    configured = list(
        dict.fromkeys(
            model for model in _configured_models() + vision_models() if model
        )
    )
    try:
        available = {model.id for model in client.models.list().data}
    except Exception as error:
        print(f"WARNING: Could not check configured Groq models: {error}")
        return

    for model in configured:
        if model not in available:
            print(f"WARNING: Configured Groq model is not available: {model}")
