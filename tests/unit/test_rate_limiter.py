import pytest

from meme_organizer.labeling.exceptions import QuotaExhaustedError, TransientApiError
from meme_organizer.labeling.rate_limiter import RateLimiter


def test_rate_limiter_retries_transient_error() -> None:
    sleeps: list[float] = []
    now = 0.0

    def fake_clock() -> float:
        return now

    def fake_sleep(sec: float) -> None:
        nonlocal now
        sleeps.append(sec)
        now += sec

    limiter = RateLimiter(rpm=60.0, max_retries=3, base_backoff_sec=2.0, sleep_fn=fake_sleep, clock_fn=fake_clock)
    attempts = 0

    def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientApiError("429 rpm")
        return "ok"

    assert limiter.execute(flaky_call) == "ok"
    assert attempts == 3
    assert len(sleeps) >= 2


def test_rate_limiter_stops_immediately_on_daily_quota() -> None:
    limiter = RateLimiter(rpm=600.0, max_retries=5, sleep_fn=lambda _: None)
    attempts = 0

    def exhausted_call() -> str:
        nonlocal attempts
        attempts += 1
        raise QuotaExhaustedError("daily limit")

    with pytest.raises(QuotaExhaustedError):
        limiter.execute(exhausted_call)

    assert attempts == 1
