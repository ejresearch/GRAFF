import os
from typing import Dict, Any, Optional
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from dotenv import load_dotenv
from ..utils.logging_config import get_logger
import logging

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Configuration from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))
ANTHROPIC_TIMEOUT = int(os.getenv("ANTHROPIC_TIMEOUT", "300"))
ANTHROPIC_MAX_RETRIES = int(os.getenv("ANTHROPIC_MAX_RETRIES", "5"))
ANTHROPIC_RETRY_MIN_WAIT = int(os.getenv("ANTHROPIC_RETRY_MIN_WAIT", "4"))
ANTHROPIC_RETRY_MAX_WAIT = int(os.getenv("ANTHROPIC_RETRY_MAX_WAIT", "10"))

# Debug logging
logger.info(f"Environment check - ANTHROPIC_API_KEY present: {bool(ANTHROPIC_API_KEY)}")
logger.info(f"Environment check - ANTHROPIC_MODEL: {ANTHROPIC_MODEL}")
if ANTHROPIC_API_KEY:
    logger.info(f"API key starts with: {ANTHROPIC_API_KEY[:10]}...")

# Validate API key
if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not found in environment. LLM calls will fail.")

# Initialize Anthropic client
try:
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=ANTHROPIC_TIMEOUT,
    ) if ANTHROPIC_API_KEY else None
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {e}")
    client = None


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM is not properly configured."""
    pass


class LLMAPIError(LLMError):
    """Raised when LLM API call fails after retries."""
    pass


def check_client_configured() -> None:
    """Verify Anthropic client is properly configured."""
    if not client or not ANTHROPIC_API_KEY:
        raise LLMConfigurationError(
            "Anthropic API key not configured. Set ANTHROPIC_API_KEY in .env file."
        )


@retry(
    stop=stop_after_attempt(ANTHROPIC_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=ANTHROPIC_RETRY_MIN_WAIT,
        max=ANTHROPIC_RETRY_MAX_WAIT
    ),
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def call_anthropic_with_retry(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: Optional[int] = None
) -> str:
    """
    Call Anthropic API with automatic retry on transient failures.

    Args:
        system_prompt: System message to set context
        user_prompt: User message with the actual request
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Optional override for max_tokens

    Returns:
        The assistant's response content

    Raises:
        LLMConfigurationError: If client is not configured
        LLMAPIError: If API call fails after retries
    """
    check_client_configured()

    try:
        tokens = max_tokens or ANTHROPIC_MAX_TOKENS
        logger.debug(f"Calling Anthropic API with model={ANTHROPIC_MODEL}, temp={temperature}, max_tokens={tokens}")

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        content = response.content[0].text
        if not content:
            raise LLMAPIError("Empty response from Anthropic API")

        logger.debug(f"Received response: {len(content)} characters")
        return content

    except (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
        logger.warning(f"Transient API error (will retry): {e}")
        raise

    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        raise LLMAPIError(f"Anthropic API call failed: {str(e)}")


def call_anthropic_structured(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Call Anthropic API and parse JSON response.

    Args:
        system_prompt: System message to set context
        user_prompt: User message with the actual request
        temperature: Sampling temperature (0.0-1.0)
        json_schema: Optional schema (unused, for compatibility)
        max_tokens: Optional override for max_tokens

    Returns:
        Parsed JSON response as dictionary

    Raises:
        LLMConfigurationError: If client is not configured
        LLMAPIError: If API call or parsing fails
    """
    import json

    # Add JSON instruction to system prompt
    enhanced_system_prompt = system_prompt + "\n\nRespond with valid JSON only. No markdown, no explanation."

    # Call API
    content = call_anthropic_with_retry(
        system_prompt=enhanced_system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Parse JSON - strip markdown code fences if present
    try:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        logger.debug("Successfully parsed JSON response")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response content: {content[:500]}...")
        raise LLMAPIError(f"Invalid JSON response from Anthropic: {str(e)}")


def get_model_info() -> Dict[str, Any]:
    """Get information about the configured Anthropic model."""
    return {
        "model": ANTHROPIC_MODEL,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "timeout": ANTHROPIC_TIMEOUT,
        "max_retries": ANTHROPIC_MAX_RETRIES,
        "configured": client is not None
    }
