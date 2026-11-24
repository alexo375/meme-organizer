import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Итоги нормализации форматов изображений в директории."""

    scanned_images: int
    converted_count: int
    errors_count: int


class BronzeNormalizer:
    """Конвертирует замаскированные HEIF/WEBP-изображения с неверным расширением в честный JPEG без потерь файлов."""

    IMAGE_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
    VALID_PAIRS: ClassVar[dict[str, frozenset[str]]] = {
        "JPEG": frozenset({".jpg", ".jpeg"}),
        "PNG": frozenset({".png"}),
        "WEBP": frozenset({".webp"}),
    }

    def __init__(self, *, jpeg_quality: int = 95) -> None:
        self._jpeg_quality = jpeg_quality

    def normalize_dir(self, root_dir: Path) -> NormalizationResult:
        """Проверяет все изображения в директории и безопасно конвертирует несовпадающие форматы в .jpg."""
        if not root_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {root_dir}")

        scanned = 0
        converted = 0
        errors = 0

        for path in sorted(root_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            scanned += 1
            try:
                if self._needs_conversion(path):
                    self._convert_to_jpeg_safely(path)
                    converted += 1
            except (OSError, UnidentifiedImageError) as exc:
                logger.warning("Не удалось нормализовать %s: %s", path.name, exc)
                errors += 1

        return NormalizationResult(
            scanned_images=scanned,
            converted_count=converted,
            errors_count=errors,
        )

    @classmethod
    def _needs_conversion(cls, path: Path) -> bool:
        with Image.open(path) as img:
            real_fmt = (img.format or "").upper()
        allowed_exts = cls.VALID_PAIRS.get(real_fmt)
        return allowed_exts is None or path.suffix.lower() not in allowed_exts

    def _convert_to_jpeg_safely(self, source_path: Path) -> Path:
        target_path = self._pick_target_path(source_path)
        tmp_path = source_path.with_name(f".{source_path.stem}.norm_tmp.jpg")

        try:
            with Image.open(source_path) as img:
                rgb = img.convert("RGB")
                rgb.save(tmp_path, format="JPEG", quality=self._jpeg_quality, optimize=True)

            with Image.open(tmp_path) as check_img:
                check_img.verify()

            tmp_path.replace(target_path)
            if target_path.resolve() != source_path.resolve() and source_path.exists():
                source_path.unlink()

            logger.info("Нормализован файл: %s -> %s", source_path.name, target_path.name)
            return target_path
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _pick_target_path(source_path: Path) -> Path:
        candidate = source_path.with_suffix(".jpg")
        if candidate.resolve() == source_path.resolve() or not candidate.exists():
            return candidate

        counter = 2
        while True:
            alt = source_path.with_name(f"{source_path.stem}-{counter}.jpg")
            if not alt.exists():
                return alt
            counter += 1
