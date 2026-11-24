import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from PIL import Image, UnidentifiedImageError

from meme_organizer.schemas.curation import CuratedRecord
from meme_organizer.schemas.labeling import MediaType, SilverRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    """Пара подозрительных дубликатов с метрикой схожести."""

    source_hash: str
    target_hash: str
    source_filename: str
    target_filename: str
    media_type: MediaType
    reason: str
    distance: int


class MediaDeduplicator:
    """Детектирует и объединяет перцептивные дубликаты изображений и видео."""

    IMAGE_HASH_SIZE: ClassVar[int] = 8
    MAX_IMAGE_HAMMING_DIST: ClassVar[int] = 4
    VIDEO_DURATION_TOLERANCE_MS: ClassVar[int] = 600

    def __init__(self, silver_dir: Path) -> None:
        self._silver_dir = silver_dir

    def find_duplicates(self, records: Sequence[SilverRecord | CuratedRecord]) -> list[DuplicateCandidate]:
        """Ищет пары дубликатов среди размеченных записей."""
        candidates: list[DuplicateCandidate] = []
        candidates.extend(self._find_image_duplicates(records))
        candidates.extend(self._find_video_duplicates(records))
        return candidates

    def _find_image_duplicates(self, records: Sequence[SilverRecord | CuratedRecord]) -> list[DuplicateCandidate]:
        image_records = [
            r
            for r in records
            if (r.media_type.value if hasattr(r.media_type, "value") else str(r.media_type)) in ("image", "gif")
        ]
        hashes: dict[str, str] = {}

        for rec in image_records:
            file_path = self._silver_dir / rec.silver_filename
            if not file_path.exists():
                continue
            dhash_val = self.compute_dhash(file_path)
            if dhash_val:
                hashes[rec.file_hash] = dhash_val

        found: list[DuplicateCandidate] = []
        rec_map = {r.file_hash: r for r in image_records}
        hash_keys = list(hashes.keys())

        for i in range(len(hash_keys)):
            h1 = hash_keys[i]
            for j in range(i + 1, len(hash_keys)):
                h2 = hash_keys[j]
                dist = self.hamming_distance(hashes[h1], hashes[h2])
                if dist <= self.MAX_IMAGE_HAMMING_DIST:
                    r1 = rec_map[h1]
                    r2 = rec_map[h2]
                    m_type = (
                        MediaType.IMAGE
                        if (r1.media_type.value if hasattr(r1.media_type, "value") else str(r1.media_type)) == "image"
                        else MediaType.GIF
                    )
                    found.append(
                        DuplicateCandidate(
                            source_hash=r1.file_hash,
                            target_hash=r2.file_hash,
                            source_filename=r1.silver_filename,
                            target_filename=r2.silver_filename,
                            media_type=m_type,
                            reason=f"Perceptual dHash distance={dist}",
                            distance=dist,
                        )
                    )
        return found

    def _find_video_duplicates(self, records: Sequence[SilverRecord | CuratedRecord]) -> list[DuplicateCandidate]:
        video_records = [
            r
            for r in records
            if (r.media_type.value if hasattr(r.media_type, "value") else str(r.media_type)) == "video"
            and r.duration_ms is not None
        ]
        found: list[DuplicateCandidate] = []

        for i in range(len(video_records)):
            v1 = video_records[i]
            for j in range(i + 1, len(video_records)):
                v2 = video_records[j]
                if v1.duration_ms is None or v2.duration_ms is None:
                    continue

                diff = abs(v1.duration_ms - v2.duration_ms)
                same_res = (v1.width == v2.width and v1.height == v2.height) and v1.width is not None

                if diff <= self.VIDEO_DURATION_TOLERANCE_MS and same_res:
                    found.append(
                        DuplicateCandidate(
                            source_hash=v1.file_hash,
                            target_hash=v2.file_hash,
                            source_filename=v1.silver_filename,
                            target_filename=v2.silver_filename,
                            media_type=MediaType.VIDEO,
                            reason=f"Same duration (Δ={diff}ms) & resolution ({v1.width}x{v1.height})",
                            distance=diff,
                        )
                    )
        return found

    @classmethod
    def compute_dhash(cls, path: Path) -> str | None:
        """Вычисляет разностный хеш (difference hash) для изображения."""
        try:
            with Image.open(path) as img:
                gray = img.convert("L").resize(
                    (cls.IMAGE_HASH_SIZE + 1, cls.IMAGE_HASH_SIZE),
                    Image.Resampling.LANCZOS,
                )
                raw_data = gray.get_flattened_data()
                pixels: list[float] = [float(p) for p in raw_data]  # type: ignore[arg-type]
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            logger.debug("Не удалось рассчитать dHash для %s: %s", path.name, exc)
            return None

        diff: list[bool] = []
        width = cls.IMAGE_HASH_SIZE + 1
        for row in range(cls.IMAGE_HASH_SIZE):
            for col in range(cls.IMAGE_HASH_SIZE):
                left = pixels[row * width + col]
                right = pixels[row * width + col + 1]
                diff.append(left > right)

        decimal = 0
        hex_parts: list[str] = []
        for index, val in enumerate(diff):
            if val:
                decimal += 2 ** (index % 8)
            if (index % 8) == 7:
                hex_parts.append(hex(decimal)[2:].rjust(2, "0"))
                decimal = 0

        return "".join(hex_parts)

    @staticmethod
    def hamming_distance(hex1: str, hex2: str) -> int:
        """Хэммингово расстояние между двумя 16-ричными строками одинаковой длины."""
        try:
            return bin(int(hex1, 16) ^ int(hex2, 16)).count("1")
        except ValueError:
            return 999
