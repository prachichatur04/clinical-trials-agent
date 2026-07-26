import logging
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.ctgov.client import CTGovClient
from app.exceptions import AppError, InternalError, ValidationError
from app.intent.llm_client import IntentLLMClient
from app.pipeline import run_pipeline
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse
from app.services.summary_generator import SummaryLLMClient

logger = logging.getLogger(__name__)

app = FastAPI(title="Clinical Trials Query-to-Visualization Agent")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - started_at) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    error = ValidationError(message="; ".join(_format_validation_error(e) for e in exc.errors()))
    return JSONResponse(status_code=error.status_code, content=error.to_response().model_dump(mode="json"))


def _format_validation_error(error: dict) -> str:
    # error["loc"] is e.g. ("body", "query") -- the field name a caller
    # actually recognizes, not the raw pydantic tuple/message dump.
    field = error["loc"][-1] if error["loc"] else "request"
    return f"{field}: {error['msg']}"


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_response().model_dump(mode="json"))


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error handling %s %s", request.method, request.url.path)
    error = InternalError(message="An unexpected error occurred.")
    return JSONResponse(status_code=error.status_code, content=error.to_response().model_dump(mode="json"))


def get_ctgov_client() -> CTGovClient | None:
    """None tells run_pipeline to construct (and later close) its own
    client. Overridden in tests to inject a fake."""
    return None


def get_llm_client() -> IntentLLMClient | None:
    """None tells parse_intent to construct its own IntentLLMClient (which
    falls back to heuristics if no API key is configured). Overridden in
    tests to inject a stub."""
    return None


def get_summary_llm_client() -> SummaryLLMClient | None:
    """None tells generate_summary to construct its own SummaryLLMClient
    (which returns no summary at all if no API key is configured).
    Overridden in tests to inject a stub."""
    return None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    ctgov_client: CTGovClient | None = Depends(get_ctgov_client),
    llm_client: IntentLLMClient | None = Depends(get_llm_client),
    summary_llm_client: SummaryLLMClient | None = Depends(get_summary_llm_client),
) -> QueryResponse:
    return await run_pipeline(
        request, ctgov_client=ctgov_client, llm_client=llm_client, summary_llm_client=summary_llm_client
    )


# Registered after /health, /query, /docs, /openapi.json so the demo UI at
# "/" never shadows the API routes above it.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
