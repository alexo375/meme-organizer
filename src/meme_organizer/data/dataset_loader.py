"""Загрузчик датасета для чтения записей корпуса и сплитов без утечки данных."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class DatasetLoader:
    """Загружает записи эталонного датасета и пары запрос-документ."""

    @staticmethod
    def load_corpus(gold_dataset_path: Path) -> list[dict[str, Any]]:
        """Загружает полный корпус документов из JSONL-файла.

        Args:
            gold_dataset_path: Путь к файлу эталонного датасета.

        Returns:
            Список словарей с записями корпуса.

        Raises:
            OSError: При ошибке чтения файла.
            json.JSONDecodeError: При некорректном JSON в строке.

        """
        with open(gold_dataset_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def load_split_info(splits_path: Path) -> dict[str, Any]:
        """Загружает метаданные разбиения из JSON-файла.

        Args:
            splits_path: Путь к файлу с метаданными сплитов.

        Returns:
            Словарь с хэшами корпуса и парами для train, val и test.

        Raises:
            OSError: При ошибке чтения файла.
            json.JSONDecodeError: При некорректном JSON.

        """
        with open(splits_path, encoding="utf-8") as f:
            data = json.load(f)
            return dict(data)

    @staticmethod
    def load_query_pairs(splits_path: Path, split_name: str) -> list[dict[str, str]]:
        """Загружает пары запрос-документ для указанного сплита.

        Args:
            splits_path: Путь к файлу с метаданными сплитов.
            split_name: Имя сплита (train, val или test).

        Returns:
            Список словарей с парами для оценки.

        Raises:
            KeyError: Если ключ вида '<split_name>_pairs' отсутствует в файле.

        """
        info = DatasetLoader.load_split_info(splits_path)
        pairs_key = f"{split_name}_pairs"
        if pairs_key not in info:
            raise KeyError(f"Сплит '{pairs_key}' отсутствует в {splits_path}")
        return list(info[pairs_key])
