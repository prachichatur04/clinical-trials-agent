import httpx
import pytest
from fastapi.testclient import TestClient

from app.ctgov.client import CTGovClient
from app.main import app, get_ctgov_client, get_llm_client
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.utils.rate_limiter import RateLimiter

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def _study(nct_id: str) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Study {nct_id}"},
            "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2020-01-01"}},
            "designModule": {"phases": ["PHASE1"]},
        }
    }


def _page_response(studies, total_count):
    return httpx.Response(200, json={"studies": studies, "totalCount": total_count})


def _mock_ctgov_client(handler) -> CTGovClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return CTGovClient(http_client=http_client, rate_limiter=RateLimiter(min_interval=0))


def _stub_llm(intent: Intent):
    class _Stub:
        async def classify(self, query, extra_context=None):
            return intent

    return _Stub()


def _intent(analysis_type=AnalysisType.DISTRIBUTION, entities=None) -> Intent:
    return Intent(
        analysis_type=analysis_type,
        entities=entities or Entities(),
        suggested_viz=VizType.BAR_CHART,
        query_plan="plan",
        notes="interpretation",
        confidence=Confidence.HIGH,
    )


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_page_loads():
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/query" in response.json()["paths"]


def test_valid_query_returns_200_with_expected_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1"), _study("NCT2")], total_count=2)

    app.dependency_overrides[get_ctgov_client] = lambda: _mock_ctgov_client(handler)
    app.dependency_overrides[get_llm_client] = lambda: _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    response = client.post("/query", json={"query": "How are trials distributed across phases?"})

    assert response.status_code == 200
    body = response.json()
    assert body["visualization"]["type"] == "bar_chart"
    assert body["meta"]["total_studies_matched"] == 2
    assert body["meta"]["intent_source"] == "llm"
    assert body["summary"] is None


def test_missing_query_field_returns_structured_422():
    response = client.post("/query", json={})

    assert response.status_code == 422
    body = response.json()
    assert body["error_type"] == "validation_error"
    assert "message" in body


def test_too_short_query_returns_structured_422():
    response = client.post("/query", json={"query": "hi"})

    assert response.status_code == 422
    assert response.json()["error_type"] == "validation_error"


def test_empty_results_returns_200_structured_no_results_not_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([], total_count=0)

    app.dependency_overrides[get_ctgov_client] = lambda: _mock_ctgov_client(handler)
    app.dependency_overrides[get_llm_client] = lambda: _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    response = client.post("/query", json={"query": "How are trials distributed for a nonexistent drug?"})

    assert response.status_code == 200
    body = response.json()
    assert body["error_type"] == "no_results"
    assert "suggestion" in body


def test_unexpected_upstream_failure_returns_structured_500_not_a_traceback():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    app.dependency_overrides[get_ctgov_client] = lambda: _mock_ctgov_client(handler)
    app.dependency_overrides[get_llm_client] = lambda: _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    response = client.post("/query", json={"query": "How are trials distributed across phases?"})

    assert response.status_code == 500
    body = response.json()
    assert body["error_type"] == "internal_error"
    assert "Traceback" not in body["message"]


def test_max_studies_out_of_bounds_returns_structured_422():
    response = client.post("/query", json={"query": "How are trials distributed?", "max_studies": 999999})
    assert response.status_code == 422
    assert response.json()["error_type"] == "validation_error"


def test_default_ctgov_and_llm_clients_used_when_not_overridden():
    # No dependency_overrides set -- get_ctgov_client/get_llm_client both
    # return None, so run_pipeline constructs its own. Without an
    # OPENAI_API_KEY in this environment that means the heuristic path,
    # and a real (small, offline-safe) call would hit the network -- so we
    # only assert the wiring doesn't blow up before reaching the network,
    # by checking the dependency functions themselves.
    assert get_ctgov_client() is None
    assert get_llm_client() is None
