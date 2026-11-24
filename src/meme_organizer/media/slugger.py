import re
import shutil
from pathlib import Path
from typing import ClassVar


class SlugManager:
    """Нормализует имена файлов в dash-case, разрешает коллизии и копирует/переименовывает файлы."""

    MAX_SLUG_LENGTH: ClassVar[int] = 60
    NON_ALNUM_RE: ClassVar[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

    def __init__(self, target_dir: Path, *, rename_in_place: bool = False) -> None:
        self._target_dir = target_dir
        self._rename_in_place = rename_in_place
        self._used_names: set[str] = set()
        if target_dir.exists():
            self._used_names.update(p.name.lower() for p in target_dir.iterdir() if p.is_file())

    @classmethod
    def sanitize(cls, raw_slug: str, *, fallback_hash: str) -> str:
        """Приводит произвольную строку к безопасному латинскому dash-case."""
        cleaned = cls.NON_ALNUM_RE.sub("-", raw_slug.lower()).strip("-")
        if len(cleaned) > cls.MAX_SLUG_LENGTH:
            cleaned = cleaned[: cls.MAX_SLUG_LENGTH].rstrip("-")
        if not cleaned:
            return f"meme-{fallback_hash[:8]}"
        return cleaned

    def allocate_filename(self, raw_slug: str, extension: str, *, fallback_hash: str) -> str:
        """Возвращает уникальное имя файла в формате '<slug><ext>' с учётом занятых имён."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        base = self.sanitize(raw_slug, fallback_hash=fallback_hash)

        candidate = f"{base}{ext}"
        counter = 2
        while candidate.lower() in self._used_names:
            candidate = f"{base}-{counter}{ext}"
            counter += 1

        self._used_names.add(candidate.lower())
        return candidate

    def materialize(self, source_path: Path, raw_slug: str, *, fallback_hash: str) -> Path:
        """Копирует файл в target_dir (или переименовывает на месте при rename_in_place=True)."""
        dest_dir = source_path.parent if self._rename_in_place else self._target_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = self.allocate_filename(raw_slug, source_path.suffix, fallback_hash=fallback_hash)
        dest_path = dest_dir / filename

        if self._rename_in_place:
            if source_path.resolve() != dest_path.resolve():
                source_path.rename(dest_path)
        else:
            shutil.copy2(source_path, dest_path)

        return dest_path
