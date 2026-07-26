import json
import logging
import time

from openai import AsyncOpenAI

from app.config import get_settings
from app.intent.prompt import SYSTEM_PROMPT, build_intent_json_schema
from app.schemas.intent import Intent

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class LLMUnavailableError(Exception):
    """No API key configured, so the LLM path can't even be attempted."""


class IntentLLMClient:
    """Thin wrapper over OpenAI Structured Outputs for Touch 1 (planning +
    classification). Raises LLMUnavailableError at construction time if no
    key is available, so callers can fall back to heuristics before ever
    making a network call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: AsyncOpenAI | None = None,
    ):
        resolved_key = api_key if api_key is not None else get_settings().openai_api_key
        if not resolved_key and client is None:
            raise LLMUnavailableError("OPENAI_API_KEY is not set")
        self._model = model
        self._client = client or AsyncOpenAI(api_key=resolved_key)

    async def classify(self, query: str, extra_context: str | None = None) -> Intent:
        """Classify one query into an Intent. Raises ValueError (via JSON
        decoding or pydantic validation) if the model's response doesn't
        parse -- callers are expected to catch that and retry/fall back.
        """
        user_content = query if not extra_context else f"{query}\n\n{extra_context}"
        logger.info("Touch 1 request: query=%r retry=%s", query, extra_context is not None)
        logger.debug("Touch 1 full prompt -- system: %s | user: %s", SYSTEM_PROMPT, user_content)

        started_at = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_schema", "json_schema": build_intent_json_schema()},
        )
        duration_ms = (time.monotonic() - started_at) * 1000

        raw = response.choices[0].message.content
        logger.debug("Touch 1 raw response: %s", raw)
        data = json.loads(raw)
        intent = Intent(**data)

        logger.info(
            "Touch 1 response (%.0fms): analysis_type=%s confidence=%s entities=%s",
            duration_ms,
            intent.analysis_type.value,
            intent.confidence.value,
            intent.entities.model_dump(exclude_none=True),
        )
        return intent
