from app.schemas.response import ErrorResponse, ErrorType


class AppError(Exception):
    """Base for every error the pipeline can raise. Each subclass fixes an
    `error_type`/`status_code` pair so a FastAPI exception handler can turn
    any AppError into the right structured ErrorResponse + HTTP status
    without a big if/elif chain.
    """

    error_type: ErrorType = ErrorType.INTERNAL_ERROR
    status_code: int = 500

    def __init__(self, message: str, suggestion: str | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(error_type=self.error_type, message=self.message, suggestion=self.suggestion)


class ValidationError(AppError):
    error_type = ErrorType.VALIDATION_ERROR
    status_code = 422


class ParsingError(AppError):
    """The LLM or CTGov returned a response we couldn't parse into our
    expected shape (should be rare: intent parsing has a heuristic
    fallback, so this is mainly for malformed CTGov responses)."""

    error_type = ErrorType.PARSING_ERROR
    status_code = 502


class NoResultsError(AppError):
    """Zero studies matched the query. Not a server failure -- returned as
    a 200 with a structured explanation, per the "never 500 on empty
    results" rule."""

    error_type = ErrorType.NO_RESULTS
    status_code = 200


class ApiError(AppError):
    """CTGov itself failed (after retries) or is unreachable."""

    error_type = ErrorType.API_ERROR
    status_code = 502


class UnsupportedQueryError(AppError):
    error_type = ErrorType.UNSUPPORTED_QUERY
    status_code = 422


class InternalError(AppError):
    error_type = ErrorType.INTERNAL_ERROR
    status_code = 500
