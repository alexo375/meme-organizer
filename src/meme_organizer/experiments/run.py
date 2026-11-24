"""CLI-раннер для проведения экспериментов с моделями поиска."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from meme_organizer.retrieval.adapters.dense_text import DenseTextRetrieverAdapter
from meme_organizer.retrieval.baselines.bm25 import BM25RetrieverAdapter
from meme_organizer.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Парсер CLI аргументов для запуска экспериментов."""
    parser = argparse.ArgumentParser(description="Запуск ML экспериментов поиска Meme Organizer")
    parser.add_argument(
        "--model",
        type=str,
        default="bm25",
        choices=["bm25", "rubert-tiny2", "e5-small"],
        help="Модель для запуска",
    )
    parser.add_argument(
        "--gold-dataset",
        type=Path,
        default=Path("data/gold/gold_dataset.jsonl"),
        help="Путь к верифицированному gold датасету",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/splits_v1.json"),
        help="Путь к файлу сплита",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        choices=["val", "test"],
        help="На каком сплите производить оценку (val или test)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Кастомное имя рана в MLflow",
    )
    return parser


def main() -> None:
    """Точка входа запуска экспериментов."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()

    adapter: Any
    if args.model == "bm25":
        adapter = BM25RetrieverAdapter(model_name="bm25-baseline")
    elif args.model == "rubert-tiny2":
        adapter = DenseTextRetrieverAdapter(
            model_name="rubert-tiny2",
            hf_model_id="cointegrated/rubert-tiny2",
            max_length=64,
        )
    elif args.model == "e5-small":
        adapter = DenseTextRetrieverAdapter(
            model_name="multilingual-e5-small",
            hf_model_id="intfloat/multilingual-e5-small",
            max_length=64,
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
    else:
        raise ValueError(f"Неизвестная модель: {args.model}")

    pipeline = RetrievalPipeline()
    metrics, run_id = pipeline.run(
        adapter=adapter,
        gold_dataset_path=args.gold_dataset,
        split_path=args.split,
        split_name=args.split_name,
        run_name=args.run_name,
    )

    logger.info("=" * 50)
    logger.info("Эксперимент завершен успешно! MLflow Run ID: %s", run_id)
    logger.info("Recall@1: %.4f", metrics.recall_at_1)
    logger.info("Recall@5: %.4f", metrics.recall_at_5)
    logger.info("MRR@5:    %.4f", metrics.mrr_at_5)
    logger.info("NDCG@5:   %.4f", metrics.ndcg_at_5)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
