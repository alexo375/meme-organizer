import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from meme_organizer.labeling.exceptions import QuotaExhaustedError, TransientApiError

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RateLimiter:
    """Ограничитель частоты запросов (RPM) с exponential backoff и jitter на временных ошибках."""

    def __init__(
        self,
        *,
        rpm: float = 8.0,
        max_retries: int = 4,
        base_backoff_sec: float = 4.0,
        max_backoff_sec: float = 90.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval_sec = 60.0 / rpm
        self._max_retries = max_retries
        self._base_backoff_sec = base_backoff_sec
        self._max_backoff_sec = max_backoff_sec
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._last_call_ts: float | None = None

    def wait_for_slot(self) -> None:
        """Выдерживает минимальный интервал между запросами согласно заданному RPM."""
        now = self._clock()
        if self._last_call_ts is not None:
            elapsed = now - self._last_call_ts
            remaining = self._min_interval_sec - elapsed
            if remaining > 0.0:
                self._sleep(remaining)
        self._last_call_ts = self._clock()

    def execute(self, fn: Callable[[], T]) -> T:
        """Выполняет функцию с учётом RPM и ретраями при TransientApiError."""
        attempt = 0
        while True:
            self.wait_for_slot()
            try:
                return fn()
            except QuotaExhaustedError:
                raise
            except TransientApiError as exc:
                attempt += 1
                if attempt > self._max_retries:
                    raise
                delay = self._compute_backoff(attempt)
                logger.warning(
                    "Временная ошибка API (%s). Повтор %d/%d через %.1f сек...",
                    exc,
                    attempt,
                    self._max_retries,
                    delay,
                )
                self._sleep(delay)

    def _compute_backoff(self, attempt: int) -> float:
        raw_delay = float(self._base_backoff_sec * (2.0 ** (attempt - 1)))
        capped = min(raw_delay, self._max_backoff_sec)
        jitter = random.uniform(0.0, capped * 0.25)  # noqa: S311
        return capped + jitter
