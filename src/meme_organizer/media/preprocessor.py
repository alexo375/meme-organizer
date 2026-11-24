import logging
import mimetypes
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

from meme_organizer.schemas.labeling import MediaFileInfo, MediaType

register_heif_opener()
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    """Подготовленный файл для отправки в VLM-API (оригинал либо временная сжатая копия)."""

    path: Path
    mime_type: str
    size_bytes: int
    is_temporary: bool


class MediaPreprocessor:
    """Готовит фото, GIF и видео для отправки в Gemini без модификации оригиналов."""

    MIME_FALLBACKS: ClassVar[dict[str, str]] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
    }

    def __init__(
        self,
        *,
        image_max_side: int = 1280,
        image_max_mb: float = 4.0,
        video_max_mb: float = 12.0,
        video_target_height: int = 480,
        ffmpeg_bin: str | None = None,
    ) -> None:
        self._image_max_side = image_max_side
        self._image_max_bytes = int(image_max_mb * 1024 * 1024)
        self._video_max_bytes = int(video_max_mb * 1024 * 1024)
        self._video_target_height = video_target_height
        self._ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")

    @contextmanager
    def prepare(self, info: MediaFileInfo) -> Iterator[PreparedMedia]:
        """Возвращает PreparedMedia и гарантированно удаляет временный файл при выходе."""
        prepared = self._build_prepared(info)
        try:
            yield prepared
        finally:
            if prepared.is_temporary and prepared.path.exists():
                prepared.path.unlink(missing_ok=True)

    def _build_prepared(self, info: MediaFileInfo) -> PreparedMedia:
        if info.media_type == MediaType.IMAGE:
            return self._prepare_image(info)
        if info.media_type == MediaType.VIDEO:
            return self._prepare_video(info)
        return PreparedMedia(
            path=info.original_path,
            mime_type=self._guess_mime(info.original_path),
            size_bytes=info.size_bytes,
            is_temporary=False,
        )

    def _prepare_image(self, info: MediaFileInfo) -> PreparedMedia:
        real_format = self._detect_image_format(info.original_path)
        expected_mime = self._guess_mime(info.original_path)
        format_mismatch = real_format not in {"JPEG", "PNG", "WEBP"} or (
            (real_format == "JPEG" and expected_mime != "image/jpeg")
            or (real_format == "PNG" and expected_mime != "image/png")
            or (real_format == "WEBP" and expected_mime != "image/webp")
        )
        needs_resize = (
            format_mismatch
            or info.size_bytes > self._image_max_bytes
            or (info.width is not None and info.width > self._image_max_side)
            or (info.height is not None and info.height > self._image_max_side)
        )
        if not needs_resize:
            return PreparedMedia(
                path=info.original_path,
                mime_type=expected_mime,
                size_bytes=info.size_bytes,
                is_temporary=False,
            )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with Image.open(info.original_path) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((self._image_max_side, self._image_max_side))
            rgb.save(tmp_path, format="JPEG", quality=88, optimize=True)

        return PreparedMedia(
            path=tmp_path,
            mime_type="image/jpeg",
            size_bytes=tmp_path.stat().st_size,
            is_temporary=True,
        )

    @staticmethod
    def _detect_image_format(path: Path) -> str:
        try:
            with Image.open(path) as img:
                return (img.format or "").upper()
        except (OSError, UnidentifiedImageError):
            return ""

    def _prepare_video(self, info: MediaFileInfo) -> PreparedMedia:
        is_webm = info.original_path.suffix.lower() == ".webm"
        needs_transcode = info.size_bytes > self._video_max_bytes or is_webm
        if not needs_transcode or not self._ffmpeg_bin:
            return PreparedMedia(
                path=info.original_path,
                mime_type=self._guess_mime(info.original_path),
                size_bytes=info.size_bytes,
                is_temporary=False,
            )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        scale_filter = f"scale=-2:'min({self._video_target_height},ih)'"
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-i",
            str(info.original_path),
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-b:a",
            "64k",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)  # noqa: S603
        except (subprocess.SubprocessError, OSError) as exc:
            tmp_path.unlink(missing_ok=True)
            logger.warning("ffmpeg сжатие не удалось для %s, отправляем оригинал: %s", info.original_path.name, exc)
            return PreparedMedia(
                path=info.original_path,
                mime_type=self._guess_mime(info.original_path),
                size_bytes=info.size_bytes,
                is_temporary=False,
            )

        return PreparedMedia(
            path=tmp_path,
            mime_type="video/mp4",
            size_bytes=tmp_path.stat().st_size,
            is_temporary=True,
        )

    @classmethod
    def _guess_mime(cls, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in cls.MIME_FALLBACKS:
            return cls.MIME_FALLBACKS[suffix]
        guessed, _ = mimetypes.guess_type(str(path))
        return guessed or "application/octet-stream"
