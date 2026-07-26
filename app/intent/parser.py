import logging
from dataclasses import dataclass

from app.intent.heuristics import classify_heuristically
from app.intent.llm_client import IntentLLMClient, LLMUnavailableError
from app.schemas.intent import Intent
from app.schemas.request import MAX_YEAR, MIN_YEAR, QueryRequest
from app.schemas.response import IntentSource

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2  # first attempt + one retry, before falling back to heuristics


class IntentSanityError(Exception):
    """An otherwise schema-valid Intent failed a cross-field sanity check
    (year ordering, plausible year bounds). Comparison-without-both-entities
    is deliberately NOT checked here -- Intent's own model validator already
    self-heals that case, so it never reaches this as a failure."""


@dataclass
class ParsedIntent:
    intent: Intent
    source: IntentSource


async def parse_intent(request: QueryRequest, llm_client: IntentLLMClient | None = None) -> ParsedIntent:
    """Touch 1 entry point: LLM plan+classify, with a validation ladder that
    degrades gracefully instead of ever failing the whole query.

    1. No API key (or no llm_client supplied and none constructible)
       -> heuristic fallback immediately.
    2. LLM call succeeds and passes cross-field sanity -> use it.
    3. LLM call fails (malformed response, sanity check, or any other
       exception from the API itself) -> one retry with the error fed back
       into the prompt.
    4. Retry also fails -> heuristic fallback.

    Any failure of the LLM path -- not just the ones the plan calls out --
    is treated as "fall back to heuristics", not "fail the request": a
    transient OpenAI outage shouldn't turn into a 500 when a valid, if
    lower-confidence, answer is available.
    """
    try:
        client = llm_client if llm_client is not None else IntentLLMClient()
    except LLMUnavailableError:
        logger.info("no LLM available, using heuristic fallback")
        return ParsedIntent(classify_heuristically(request), IntentSource.HEURISTIC_FALLBACK)

    retry_context: str | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            intent = await client.classify(request.query, extra_context=retry_context)
            _check_sane(intent)
        except Exception as error:  # noqa: BLE001 -- any LLM-path failure degrades, doesn't raise
            logger.warning("intent parsing attempt %d failed: %s", attempt, error)
            retry_context = f"Your previous response was invalid ({error}). Correct it and respond again."
            continue
        source = IntentSource.LLM if attempt == 0 else IntentSource.LLM_RETRY
        return ParsedIntent(intent, source)

    logger.warning("LLM intent parsing exhausted retries, using heuristic fallback")
    return ParsedIntent(classify_heuristically(request), IntentSource.HEURISTIC_FALLBACK)


def _check_sane(intent: Intent) -> None:
    start, end = intent.entities.start_year, intent.entities.end_year
    for year in (start, end):
        if year is not None and not (MIN_YEAR <= year <= MAX_YEAR):
            raise IntentSanityError(f"year {year} is out of plausible range [{MIN_YEAR}, {MAX_YEAR}]")
    if start is not None and end is not None and start > end:
        raise IntentSanityError(f"start_year {start} > end_year {end}")
