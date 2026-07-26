import asyncio
import time
from collections.abc import Awaitable, Callable


class RateLimiter:
    """Paces requests under CTGov's ~50 req/min limit and backs off on 429s.

    `clock` and `sleep` are injectable so tests can control timing without
    monkeypatching the stdlib clock/sleep that asyncio's own internals rely on.
    """

    def __init__(
        self,
        min_interval: float = 1.2,
        base_backoff: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.min_interval = min_interval
        self.base_backoff = base_backoff
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    async def wait(self) -> None:
        """Sleep just long enough to keep successive requests min_interval apart."""
        now = self._clock()
        if self._last_request_at is not None:
            remaining = self.min_interval - (now - self._last_request_at)
            if remaining > 0:
                await self._sleep(remaining)
        self._last_request_at = now

    def backoff_delay(self, attempt: int) -> float:
        """Exponential backoff delay in seconds for a 429 retry (attempt is 0-indexed)."""
        return self.base_backoff * (2**attempt)
