import os
from typing import Dict, Any, Optional
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
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

# Load environment variables from .env file (optional - Render uses actual env vars)
load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "16000"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "300"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
OPENAI_RETRY_MIN_WAIT = int(os.getenv("OPENAI_RETRY_MIN_WAIT", "4"))
OPENAI_RETRY_MAX_WAIT = int(os.getenv("OPENAI_RETRY_MAX_WAIT", "10"))

# Debug logging for environment variables (helpful for deployment troubleshooting)
logger.info(f"Environment check - OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")
logger.info(f"Environment check - OPENAI_MODEL: {OPENAI_MODEL}")
logger.info(f"Environment check - USE_ACTUAL_LLM: {os.getenv('USE_ACTUAL_LLM', 'not set')}")
if OPENAI_API_KEY:
    logger.info(f"API key starts with: {OPENAI_API_KEY[:10]}...")

# Validate API key
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment. LLM calls will fail.")
    logger.warning("Set OPENAI_API_KEY as an environment variable or in .env file.")

# Initialize OpenAI client
try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT,
    ) if OPENAI_API_KEY else None
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    logger.error("This may be due to version incompatibility. Try: pip install 'httpx<0.28'")
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
    """
    Verify OpenAI client is properly configured.

    Raises:
        LLMConfigurationError: If API key is missing
    """
    if not client or not OPENAI_API_KEY:
        raise LLMConfigurationError(
            "OpenAI API key not configured. Set OPENAI_API_KEY in .env file."
        )


@retry(
    stop=stop_after_attempt(OPENAI_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=OPENAI_RETRY_MIN_WAIT,
        max=OPENAI_RETRY_MAX_WAIT
    ),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def call_openai_with_retry(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    response_format: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Call OpenAI API with automatic retry on transient failures.
    Uses the Responses API with gpt-5.2.

    Args:
        system_prompt: System message to set context
        user_prompt: User message with the actual request
        temperature: Sampling temperature (0.0-2.0)
        response_format: Optional JSON schema for structured output
        max_tokens: Optional override for max_tokens (default: OPENAI_MAX_TOKENS)

    Returns:
        The assistant's response content

    Raises:
        LLMConfigurationError: If client is not configured
        LLMAPIError: If API call fails after retries
    """
    check_client_configured()

    try:
        tokens = max_tokens or OPENAI_MAX_TOKENS
        logger.debug(f"Calling OpenAI API with model={OPENAI_MODEL}, temp={temperature}, max_tokens={tokens}")

        # Combine system and user prompts for Responses API
        combined_input = f"{system_prompt}\n\n---\n\n{user_prompt}"

        # Add JSON instruction if response_format is specified
        if response_format:
            combined_input += "\n\nRespond with valid JSON only."

        # Use Responses API for gpt-5.2
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=combined_input
        )

        content = response.output_text
        if not content:
            raise LLMAPIError("Empty response from OpenAI API")

        logger.debug(f"Received response: {len(content)} characters")
        return content

    except (RateLimitError, APITimeoutError, APIConnectionError) as e:
        # These will be retried by tenacity
        logger.warning(f"Transient API error (will retry): {e}")
        raise

    except Exception as e:
        # Other errors won't be retried
        logger.error(f"OpenAI API error: {e}")
        raise LLMAPIError(f"OpenAI API call failed: {str(e)}")


def call_openai_structured(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Call OpenAI API and parse JSON response.

    Args:
        system_prompt: System message to set context
        user_prompt: User message with the actual request
        temperature: Sampling temperature (0.0-2.0)
        json_schema: Optional Pydantic model schema for validation
        max_tokens: Optional override for max_tokens (default: OPENAI_MAX_TOKENS)

    Returns:
        Parsed JSON response as dictionary

    Raises:
        LLMConfigurationError: If client is not configured
        LLMAPIError: If API call or parsing fails
    """
    import json

    # Request JSON response format
    response_format = {"type": "json_object"} if json_schema else None

    # Add JSON instruction to system prompt
    enhanced_system_prompt = system_prompt
    if json_schema:
        enhanced_system_prompt += "\n\nRespond with valid JSON matching the provided schema."

    # Call API
    content = call_openai_with_retry(
        system_prompt=enhanced_system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        response_format=response_format,
        max_tokens=max_tokens
    )

    # Parse JSON - strip markdown code fences if present
    try:
        # Remove markdown code fences if present
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]  # Remove ```json
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]  # Remove ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]  # Remove trailing ```
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        logger.debug("Successfully parsed JSON response")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response content: {content[:500]}...")
        raise LLMAPIError(f"Invalid JSON response from OpenAI: {str(e)}")


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the configured OpenAI model.

    Returns:
        Dictionary with model configuration details
    """
    return {
        "model": OPENAI_MODEL,
        "max_tokens": OPENAI_MAX_TOKENS,
        "timeout": OPENAI_TIMEOUT,
        "max_retries": OPENAI_MAX_RETRIES,
        "configured": client is not None
    }
