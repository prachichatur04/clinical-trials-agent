import httpx
import pytest

from app.ctgov.client import CTGovClient
from app.utils.rate_limiter import RateLimiter


def _make_client(handler, sleep_calls=None):
    """A CTGovClient wired to a mock transport and a rate limiter that never
    actually waits, so tests run instantly regardless of the real clock."""
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    rate_limiter = RateLimiter(min_interval=0, sleep=_noop_sleep)
    return CTGovClient(
        http_client=http_client,
        rate_limiter=rate_limiter,
        sleep=_recording_sleep(sleep_calls) if sleep_calls is not None else _noop_sleep,
    )


def _page_response(studies, total_count, next_page_token=None):
    body = {"studies": studies, "totalCount": total_count}
    if next_page_token:
        body["nextPageToken"] = next_page_token
    return httpx.Response(200, json=body)


async def test_search_returns_single_page():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["countTotal"] == "true"
        return _page_response([{"nctId": "NCT1"}], total_count=1)

    client = _make_client(handler)
    result = await client.search(drug_name="Pembrolizumab")

    assert result["totalCount"] == 1
    assert result["studies"] == [{"nctId": "NCT1"}]


async def test_paginate_stops_when_no_next_page_token():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("pageToken"))
        return _page_response([{"nctId": "NCT1"}, {"nctId": "NCT2"}], total_count=2)

    client = _make_client(handler)
    studies, total = await client.paginate(max_studies=500, drug_name="x")

    assert len(calls) == 1
    assert total == 2
    assert len(studies) == 2


async def test_paginate_follows_next_page_token_across_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("pageToken")
        if not token:
            return _page_response([{"nctId": "NCT1"}], total_count=3, next_page_token="page2")
        elif token == "page2":
            return _page_response([{"nctId": "NCT2"}], total_count=3, next_page_token="page3")
        else:
            return _page_response([{"nctId": "NCT3"}], total_count=3)

    client = _make_client(handler)
    studies, total = await client.paginate(max_studies=500, drug_name="x")

    assert [s["nctId"] for s in studies] == ["NCT1", "NCT2", "NCT3"]
    assert total == 3


async def test_paginate_stops_at_max_studies_cap_even_with_more_pages_available():
    def handler(request: httpx.Request) -> httpx.Response:
        # Always claims there's another page, to prove the cap -- not the
        # server -- ends the loop.
        return _page_response([{"nctId": "NCT1"}], total_count=10_000, next_page_token="more")

    client = _make_client(handler)
    studies, total = await client.paginate(max_studies=3, drug_name="x")

    assert len(studies) == 3
    assert total == 10_000


async def test_paginate_requests_shrinking_page_size_near_cap():
    requested_sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_sizes.append(int(request.url.params["pageSize"]))
        token = request.url.params.get("pageToken")
        if not token:
            return _page_response([{"nctId": f"NCT{i}"} for i in range(2)], total_count=100, next_page_token="p2")
        return _page_response([{"nctId": "NCT99"}], total_count=100)

    client = _make_client(handler)
    await client.paginate(max_studies=3, page_size=2, drug_name="x")

    # first page asked for 2 (remaining=3), second page only 1 remained
    assert requested_sizes == [2, 1]


async def test_paginate_stops_early_if_server_returns_empty_studies():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([], total_count=0, next_page_token="ghost")

    client = _make_client(handler)
    studies, total = await client.paginate(max_studies=500, drug_name="x")

    assert studies == []
    assert total == 0


async def test_retries_once_on_429_then_succeeds():
    sleep_calls = []
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429)
        return _page_response([{"nctId": "NCT1"}], total_count=1)

    client = _make_client(handler, sleep_calls=sleep_calls)
    result = await client.search(drug_name="x")

    assert attempts["count"] == 2
    assert result["totalCount"] == 1
    assert sleep_calls == [pytest.approx(1.0)]  # backoff_delay(attempt=0) with default base_backoff


async def test_raises_after_exhausting_retries_on_repeated_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    client = _make_client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.search(drug_name="x")


async def test_non_429_error_raises_immediately_without_retrying():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(500)

    client = _make_client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.search(drug_name="x")

    assert attempts["count"] == 1


def _noop_sleep(seconds):
    async def _inner():
        return None

    return _inner()


def _recording_sleep(calls):
    async def _fake_sleep(seconds):
        calls.append(seconds)

    return _fake_sleep
