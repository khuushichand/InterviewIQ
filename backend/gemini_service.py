"""Reusable Gemini service with retry logic, 503 error handling, and offline fallbacks."""
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend/.env is loaded
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

_client = None
if API_KEY:
    try:
        from google import genai
        _client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"[GeminiService] Initialization warning: {e}")

def get_client():
    return _client

def generate_content_with_retry(
    contents: str,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    model: str | None = None
) -> str | None:
    """
    Invokes Gemini with automatic exponential backoff for temporary errors (e.g., 503 UNAVAILABLE, 429 rate limit).
    Returns the response text or None if all retries fail.
    """
    client = get_client()
    if not client:
        return None

    from google.genai import types
    from google.genai.errors import ServerError, APIError

    target_model = model or MODEL_NAME
    config_params = {}
    if system_instruction:
        config_params["system_instruction"] = system_instruction
    if response_mime_type:
        config_params["response_mime_type"] = response_mime_type

    config = types.GenerateContentConfig(**config_params) if config_params else None

    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config
            )
            if res and res.text:
                return res.text.strip()
            return None
        except (ServerError, APIError) as e:
            last_error = e
            status_code = getattr(e, "code", getattr(e, "status_code", None))
            # Handle 503 (Model unavailable/high demand) and 429 (Rate limit)
            print(f"[GeminiService] Attempt {attempt + 1}/{max_retries} encountered API error ({status_code}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                print(f"[GeminiService] Exhausted retries for model {target_model}: {last_error}")
        except Exception as e:
            last_error = e
            print(f"[GeminiService] Unexpected error during content generation: {e}")
            break

    return None

