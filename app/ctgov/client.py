import asyncio
from collections.abc import Awaitable, Callable

import httpx

from app.ctgov.query_builder import build_query_params
from app.utils.rate_limiter import RateLimiter

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2"
DEFAULT_PAGE_SIZE = 1000
MAX_STUDIES_HARD_CAP = 5000
MAX_RETRIES = 4


class CTGovClient:
    """Async wrapper over the ClinicalTrials.gov v2 /studies endpoint.

    `sleep` backs the 429 backoff delay and is injectable for tests, same
    reasoning as RateLimiter: don't monkeypatch the stdlib clock/sleep that
    asyncio's own internals depend on.
    """

    def __init__(
        self,
        base_url: str = CTGOV_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_retries: int = MAX_RETRIES,
    ):
        self._base_url = base_url
        self._http = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_http_client = http_client is None
        self._rate_limiter = rate_limiter or RateLimiter()
        self._sleep = sleep
        self._max_retries = max_retries

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    async def search(self, **query_kwargs) -> dict:
        """Fetch a single raw page: {"studies": [...], "totalCount": int, "nextPageToken": str | None}."""
        params = build_query_params(**query_kwargs)
        return await self._get_with_retry(params)

    async def paginate(self, *, max_studies: int = 500, **query_kwargs) -> tuple[list[dict], int]:
        """Fetch up to `max_studies` raw study dicts (hard-capped at 5000).

        Returns (studies, total_count_matched) -- total_count_matched is the
        server's full match count (from countTotal=true), which may exceed
        len(studies) if the cap was hit.
        """
        page_size = min(query_kwargs.pop("page_size", DEFAULT_PAGE_SIZE), DEFAULT_PAGE_SIZE)
        cap = min(max_studies, MAX_STUDIES_HARD_CAP)

        all_studies: list[dict] = []
        total_count = 0
        page_token: str | None = None

        while len(all_studies) < cap:
            remaining = cap - len(all_studies)
            params = build_query_params(
                **query_kwargs,
                page_size=min(page_size, remaining),
                page_token=page_token,
            )
            page = await self._get_with_retry(params)
            total_count = page.get("totalCount", total_count)
            studies = page.get("studies", [])
            all_studies.extend(studies)

            page_token = page.get("nextPageToken")
            if not page_token or not studies:
                break

        return all_studies, total_count

    async def _get_with_retry(self, params: dict) -> dict:
        response: httpx.Response | None = None
        for attempt in range(self._max_retries):
            await self._rate_limiter.wait()
            response = await self._http.get(f"{self._base_url}/studies", params=params)
            if response.status_code == 429 and attempt < self._max_retries - 1:
                await self._sleep(self._rate_limiter.backoff_delay(attempt))
                continue
            response.raise_for_status()
            return response.json()

        # Retries exhausted on repeated 429s.
        response.raise_for_status()
        raise httpx.HTTPStatusError("exhausted retries", request=response.request, response=response)
