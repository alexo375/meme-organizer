import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

from meme_organizer.schemas.labeling import MediaFileInfo, MediaType

register_heif_opener()
logger = logging.getLogger(__name__)


class MediaScanner:
    """Сканирует директорию с мемами, вычисляет SHA-256 и извлекает базовые метаданные."""

    IMAGE_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
    GIF_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".gif"})
    VIDEO_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".mp4", ".webm", ".mov", ".mkv"})
    HASH_CHUNK_BYTES: ClassVar[int] = 1024 * 1024

    def __init__(self, *, ffprobe_bin: str | None = None) -> None:
        self._ffprobe_bin = ffprobe_bin or shutil.which("ffprobe")

    def scan(self, root_dir: Path) -> list[MediaFileInfo]:
        """Возвращает отсортированный список поддерживаемых медиафайлов из директории."""
        if not root_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {root_dir}")

        results: list[MediaFileInfo] = []
        for path in sorted(root_dir.rglob("*")):
            if not path.is_file():
                continue
            media_type = self.detect_media_type(path)
            if media_type is None:
                continue
            results.append(self.inspect_file(path, media_type))
        return results

    @classmethod
    def detect_media_type(cls, path: Path) -> MediaType | None:
        """Определяет MediaType по расширению файла или возвращает None."""
        suffix = path.suffix.lower()
        if suffix in cls.IMAGE_EXTENSIONS:
            return MediaType.IMAGE
        if suffix in cls.GIF_EXTENSIONS:
            return MediaType.GIF
        if suffix in cls.VIDEO_EXTENSIONS:
            return MediaType.VIDEO
        return None

    @classmethod
    def compute_sha256(cls, path: Path) -> str:
        """Вычисляет SHA-256 файла потоково."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(cls.HASH_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()

    def inspect_file(self, path: Path, media_type: MediaType) -> MediaFileInfo:
        """Собирает метаданные одного файла (размер, хеш, разрешение, длительность)."""
        file_hash = self.compute_sha256(path)
        size_bytes = path.stat().st_size

        if media_type == MediaType.VIDEO:
            duration_ms, width, height = self._probe_video(path)
        else:
            duration_ms = None
            width, height = self._probe_image(path)

        return MediaFileInfo(
            original_path=path,
            file_hash=file_hash,
            media_type=media_type,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            width=width,
            height=height,
        )

    @staticmethod
    def _probe_image(path: Path) -> tuple[int | None, int | None]:
        try:
            with Image.open(path) as img:
                return img.width, img.height
        except (OSError, UnidentifiedImageError) as exc:
            logger.warning("Не удалось прочитать размеры изображения %s: %s", path.name, exc)
            return None, None

    def _probe_video(self, path: Path) -> tuple[int | None, int | None, int | None]:
        if not self._ffprobe_bin:
            return None, None, None

        cmd = [
            self._ffprobe_bin,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)  # noqa: S603
            payload = json.loads(proc.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            logger.warning("ffprobe не смог разобрать %s: %s", path.name, exc)
            return None, None, None

        duration_raw = payload.get("format", {}).get("duration")
        duration_ms = int(float(duration_raw) * 1000) if duration_raw else None

        width: int | None = None
        height: int | None = None
        for stream in payload.get("streams", []):
            if stream.get("codec_type") == "video":
                width = stream.get("width")
                height = stream.get("height")
                break

        return duration_ms, width, height
