import logging
import time
from typing import Any, ClassVar, Protocol

import httpx2
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import ValidationError

from meme_organizer.labeling.exceptions import (
    LabelingError,
    QuotaExhaustedError,
    SafetyBlockedError,
    TransientApiError,
)
from meme_organizer.labeling.prompts import PromptRegistry
from meme_organizer.media.preprocessor import PreparedMedia
from meme_organizer.schemas.labeling import MediaFileInfo, MemeLabelResponse

logger = logging.getLogger(__name__)


class MediaLabeler(Protocol):
    """Интерфейс VLM-разметчика для поддержки разных провайдеров и моков в тестах."""

    @property
    def model_name(self) -> str:
        """Имя используемой модели."""
        ...

    def label(self, info: MediaFileInfo, prepared: PreparedMedia) -> MemeLabelResponse:
        """Выполняет структурированную разметку одного медиафайла."""
        ...


class GeminiLabeler:
    """Клиент разметки мемов через Google Gemini API (Structured Output + Files API)."""

    SAFETY_CATEGORIES: ClassVar[tuple[genai_types.HarmCategory, ...]] = (
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
    TRANSIENT_STATUS_CODES: ClassVar[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
    DAILY_QUOTA_MARKERS: ClassVar[tuple[str, ...]] = ("per day", "daily", "rpd", "billing", "free_tier_requests")

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        inline_max_mb: float = 10.0,
        file_active_timeout_sec: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY не задан. Укажи его в файле .env или переменной окружения.")
        self._model_name = model_name
        self._inline_max_bytes = int(inline_max_mb * 1024 * 1024)
        self._file_active_timeout_sec = file_active_timeout_sec
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1beta"),
        )

    @property
    def model_name(self) -> str:
        """Имя используемой модели Gemini."""
        return self._model_name

    def label(self, info: MediaFileInfo, prepared: PreparedMedia) -> MemeLabelResponse:
        """Отправляет медиафайл в Gemini и возвращает валидированный MemeLabelResponse."""
        prompt = PromptRegistry.build_user_prompt(info.media_type, info.original_path.name)
        config = self._build_generate_config()

        uploaded_name: str | None = None
        try:
            media_part, uploaded_name = self._prepare_content_part(prepared)
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[media_part, prompt],
                config=config,
            )
            return self._parse_response(response)
        except genai_errors.APIError as exc:
            self._handle_api_error(exc)
        except (httpx2.HTTPError, TimeoutError, ConnectionError) as exc:
            raise TransientApiError(f"Сетевая ошибка/таймаут Gemini API: {exc}") from exc
        finally:
            if uploaded_name is not None:
                self._safe_delete_remote_file(uploaded_name)

        raise LabelingError("Неожиданное завершение вызова Gemini API.")

    def _build_generate_config(self) -> genai_types.GenerateContentConfig:
        safety_settings = [
            genai_types.SafetySetting(
                category=category,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
            )
            for category in self.SAFETY_CATEGORIES
        ]
        return genai_types.GenerateContentConfig(
            system_instruction=PromptRegistry.SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=MemeLabelResponse,
            temperature=0.2,
            safety_settings=safety_settings,
        )

    def _prepare_content_part(self, prepared: PreparedMedia) -> tuple[Any, str | None]:
        if prepared.size_bytes <= self._inline_max_bytes:
            data = prepared.path.read_bytes()
            part = genai_types.Part.from_bytes(data=data, mime_type=prepared.mime_type)
            return part, None

        uploaded = self._client.files.upload(file=str(prepared.path))
        if not uploaded.name:
            raise LabelingError("Files API вернул файл без имени.")
        active_file = self._wait_file_active(uploaded.name)
        return active_file, uploaded.name

    def _wait_file_active(self, file_name: str) -> Any:
        deadline = time.monotonic() + self._file_active_timeout_sec
        while time.monotonic() < deadline:
            remote = self._client.files.get(name=file_name)
            state_name = str(getattr(remote.state, "name", remote.state)).upper()
            if state_name == "ACTIVE":
                return remote
            if state_name == "FAILED":
                raise LabelingError(f"Gemini Files API не смог обработать видео: {file_name}")
            time.sleep(2.0)
        raise TransientApiError(f"Таймаут ожидания обработки файла в Gemini Files API: {file_name}")

    def _safe_delete_remote_file(self, file_name: str) -> None:
        try:
            self._client.files.delete(name=file_name)
        except (genai_errors.APIError, OSError) as exc:
            logger.debug("Не удалось удалить временный файл %s в Files API: %s", file_name, exc)

    @staticmethod
    def _parse_response(response: Any) -> MemeLabelResponse:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            raise SafetyBlockedError(f"Запрос заблокирован фильтром безопасности: {block_reason}")

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "")).upper()
            if any(token in finish_reason for token in ("SAFETY", "PROHIBITED", "BLOCKLIST", "RECITATION")):
                raise SafetyBlockedError(f"Ответ заблокирован фильтром: {finish_reason}")

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, MemeLabelResponse):
            return parsed

        text = getattr(response, "text", None)
        if not text:
            raise SafetyBlockedError("Пустой ответ от Gemini (вероятная блокировка фильтрами).")

        try:
            return MemeLabelResponse.model_validate_json(text)
        except ValidationError as exc:
            raise LabelingError(f"Невалидный JSON от Gemini: {exc}") from exc

    @classmethod
    def _handle_api_error(cls, exc: genai_errors.APIError) -> None:
        code = getattr(exc, "code", None)
        message = str(exc)
        lower_msg = message.lower()

        if code == 429:
            if any(marker in lower_msg for marker in cls.DAILY_QUOTA_MARKERS):
                raise QuotaExhaustedError(f"Исчерпана квота Gemini API: {message}") from exc
            raise TransientApiError(f"Превышен RPM-лимит Gemini (429): {message}") from exc

        if code in cls.TRANSIENT_STATUS_CODES:
            raise TransientApiError(f"Временная ошибка Gemini API ({code}): {message}") from exc

        raise LabelingError(f"Ошибка Gemini API ({code}): {message}") from exc
