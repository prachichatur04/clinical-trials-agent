import pytest

from app.utils.rate_limiter import RateLimiter


async def test_first_wait_does_not_sleep():
    sleep_calls = []
    limiter = RateLimiter(min_interval=1.2, clock=_fake_clock([100.0]), sleep=_recording_sleep(sleep_calls))

    await limiter.wait()

    assert sleep_calls == []


async def test_second_wait_sleeps_remaining_interval():
    sleep_calls = []
    limiter = RateLimiter(
        min_interval=1.2, clock=_fake_clock([100.0, 100.3]), sleep=_recording_sleep(sleep_calls)
    )

    await limiter.wait()
    await limiter.wait()

    assert sleep_calls == [pytest.approx(0.9)]


async def test_wait_skips_sleep_when_interval_already_elapsed():
    sleep_calls = []
    limiter = RateLimiter(
        min_interval=1.2, clock=_fake_clock([100.0, 105.0]), sleep=_recording_sleep(sleep_calls)
    )

    await limiter.wait()
    await limiter.wait()

    assert sleep_calls == []


def test_backoff_delay_grows_exponentially():
    limiter = RateLimiter(base_backoff=1.0)
    assert limiter.backoff_delay(0) == 1.0
    assert limiter.backoff_delay(1) == 2.0
    assert limiter.backoff_delay(2) == 4.0
    assert limiter.backoff_delay(3) == 8.0


def test_backoff_delay_respects_custom_base():
    limiter = RateLimiter(base_backoff=0.5)
    assert limiter.backoff_delay(0) == 0.5
    assert limiter.backoff_delay(2) == 2.0


def _fake_clock(values):
    it = iter(values)
    return lambda: next(it)


def _recording_sleep(calls):
    async def _fake_sleep(seconds):
        calls.append(seconds)

    return _fake_sleep
