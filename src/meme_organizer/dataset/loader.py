"""Dataset loader reading gold records and split pairs without data leaks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class DatasetLoader:
    """Loads immutable gold dataset records and query pairs."""

    @staticmethod
    def load_corpus(gold_dataset_path: Path) -> list[dict[str, Any]]:
        """Load entire corpus of documents (never filtered or split)."""
        with open(gold_dataset_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def load_split_info(splits_path: Path) -> dict[str, Any]:
        """Load split metadata (corpus hashes and train/val/test query pairs)."""
        with open(splits_path, encoding="utf-8") as f:
            data = json.load(f)
            return dict(data)

    @staticmethod
    def load_query_pairs(splits_path: Path, split_name: str) -> list[dict[str, str]]:
        """Load query pairs for a specific split (train, val, or test)."""
        info = DatasetLoader.load_split_info(splits_path)
        pairs_key = f"{split_name}_pairs"
        if pairs_key not in info:
            raise KeyError(f"Сплит '{pairs_key}' отсутствует в {splits_path}")
        return list(info[pairs_key])
