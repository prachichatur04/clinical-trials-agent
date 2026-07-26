import logging
import os

from openai import AsyncOpenAI

from app.intent.llm_client import LLMUnavailableError
from app.schemas.intent import AnalysisType
from app.schemas.internal import AggregatedResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"

SUMMARY_SYSTEM_PROMPT = """You are the interpretation component of a clinical trials analysis agent. \
Given these analysis results, write 2-3 factual sentences summarizing the key findings. Also flag \
any data quality concerns.

Rules:
- State only what the data shows. Never speculate about causes.
- Mention the most notable finding (largest bucket, trend direction, dominant entity).
- If data was truncated or had quality issues, mention briefly."""


class SummaryLLMClient:
    """Touch 2's OpenAI wrapper. Separate from IntentLLMClient (Touch 1)
    since it's a plain free-text completion, not a structured-output call --
    but reuses the same LLMUnavailableError so both touches degrade the
    same way when no API key is configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: AsyncOpenAI | None = None,
    ):
        resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not resolved_key and client is None:
            raise LLMUnavailableError("OPENAI_API_KEY is not set")
        self._model = model
        self._client = client or AsyncOpenAI(api_key=resolved_key)

    async def summarize(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


async def generate_summary(
    query: str,
    analysis_type: AnalysisType,
    aggregated: AggregatedResult,
    total_matched: int,
    total_fetched: int,
    llm_client: SummaryLLMClient | None = None,
) -> str | None:
    """Touch 2 entry point. Never raises and never blocks the response on
    failure: no API key, or any error from the call itself, both just mean
    `summary` stays None -- this is explicitly not on the critical path,
    unlike Touch 1's intent parsing (which always has the heuristic
    fallback to produce *something*).
    """
    try:
        client = llm_client if llm_client is not None else SummaryLLMClient()
    except LLMUnavailableError:
        logger.info("no LLM available, skipping summary")
        return None

    prompt = _build_prompt(query, analysis_type, aggregated, total_matched, total_fetched)
    try:
        return await client.summarize(prompt)
    except Exception as error:  # noqa: BLE001 -- summary failures never fail the request
        logger.warning("summary generation failed: %s", error)
        return None


def _build_prompt(
    query: str,
    analysis_type: AnalysisType,
    aggregated: AggregatedResult,
    total_matched: int,
    total_fetched: int,
) -> str:
    lines = [
        f"Analysis type: {analysis_type.value}",
        f"Query: {query}",
        f"Total studies matched: {total_matched} (fetched: {total_fetched})",
    ]

    if aggregated.network is not None:
        top_edges = sorted(aggregated.network.edges, key=lambda edge: -edge.weight)[:10]
        lines.append("Top network edges (source -> target: weight):")
        lines.extend(f"  {edge.source} -> {edge.target}: {edge.weight}" for edge in top_edges)
    elif aggregated.stat_value is not None:
        lines.append(f"Total count: {aggregated.stat_value}")
    else:
        lines.append("Results (top 10 buckets):")
        for bucket in aggregated.buckets[:10]:
            series_suffix = f" [{bucket.series}]" if bucket.series else ""
            lines.append(f"  {bucket.key}{series_suffix}: {bucket.count}")

    if aggregated.assumptions:
        lines.append("Data quality notes:")
        lines.extend(f"  - {assumption}" for assumption in aggregated.assumptions)

    return "\n".join(lines)
