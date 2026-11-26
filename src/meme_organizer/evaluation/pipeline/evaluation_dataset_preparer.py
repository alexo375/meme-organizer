"""Подготовка данных корпуса и сплита для оценки ретривера."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from meme_organizer.data.dataset_loader import DatasetLoader

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class EvaluationSplitData:
    """Подготовленные данные сплита для оценки и обучения."""

    corpus_docs: list[dict[str, Any]]
    corpus_hashes: list[str]
    queries: list[str]
    ground_truth_indices: list[int]
    train_pairs: list[dict[str, Any]]
    val_pairs: list[dict[str, Any]]
    split_version: str


class EvaluationDatasetPreparer:
    """Загрузчик и подготовитель данных корпуса и сплитов."""

    @staticmethod
    def prepare(
        gold_dataset_path: Path,
        split_path: Path,
        split_name: str,
    ) -> EvaluationSplitData:
        """Загружает корпус и сплит, сопоставляя запросы с индексами документов.

        Вход: путь к gold-датасету, путь к файлу сплитов, имя сплита.
        Выход: объект EvaluationSplitData.
        Исключения: ValueError, если указанный сплит пуст.
        """
        corpus_docs = DatasetLoader.load_corpus(gold_dataset_path)
        split_info = DatasetLoader.load_split_info(split_path)

        corpus_hash_to_idx = {doc["file_hash"]: i for i, doc in enumerate(corpus_docs)}
        corpus_hashes = [doc["file_hash"] for doc in corpus_docs]

        pairs = split_info.get(f"{split_name}_pairs", [])
        if not pairs:
            raise ValueError(f"Сплит '{split_name}_pairs' пуст в {split_path}")

        queries = [p["query"] for p in pairs]
        ground_truth_indices = [corpus_hash_to_idx[p["file_hash"]] for p in pairs]

        return EvaluationSplitData(
            corpus_docs=corpus_docs,
            corpus_hashes=corpus_hashes,
            queries=queries,
            ground_truth_indices=ground_truth_indices,
            train_pairs=split_info.get("train_pairs", []),
            val_pairs=split_info.get("val_pairs", []),
            split_version=str(split_info.get("version", "v1")),
        )
