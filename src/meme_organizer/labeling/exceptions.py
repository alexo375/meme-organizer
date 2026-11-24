class LabelingError(Exception):
    """Базовое исключение пайплайна авторазметки."""


class SafetyBlockedError(LabelingError):
    """Контент заблокирован фильтрами безопасности VLM-провайдера."""


class QuotaExhaustedError(LabelingError):
    """Исчерпана дневная квота (RPD) или биллинг-лимит API."""


class TransientApiError(LabelingError):
    """Временная ошибка API (429 RPM, 500, 503, сетевой таймаут), пригодная для ретрая."""
