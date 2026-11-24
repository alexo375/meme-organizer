"""Генератор фиксированных сплитов данных (Train / Val / Test) с защитой от утечек."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryPair:
    """Пара запрос - идентификатор мема."""

    query: str
    file_hash: str
    scene_group: str


@dataclass(frozen=True)
class SplitDataset:
    """Структура фиксированного сплита данных."""

    version: str
    seed: int
    corpus_hashes: list[str]
    train_pairs: list[dict[str, str]]
    val_pairs: list[dict[str, str]]
    test_pairs: list[dict[str, str]]


class SceneGroupDetector:
    """Определяет группу сцены (scene_group) для предотвращения утечек данных."""

    @staticmethod
    def detect_group(record: dict[str, Any]) -> str:
        """Извлекает ключевую сущность мема (персонаж / фильм / сериал / шаблон)."""
        # Сначала проверяем первые теги и подпись пользователя
        tags = [t.lower().strip() for t in record.get("tags", [])]
        user_caption = (record.get("user_caption") or "").lower().strip()
        rich_desc = (record.get("rich_description") or "").lower().strip()

        # Поиск известных сущностей по тегам и подписи
        combined = f"{user_caption} {' '.join(tags)} {rich_desc}"

        known_entities = [
            ("уолтер уайт", "breaking_bad_walter"),
            ("во все тяжкие", "breaking_bad_general"),
            ("джесси пинкман", "breaking_bad_jesse"),
            ("стонкс", "stonks_guy"),
            ("багз банни", "bugs_bunny"),
            ("ди каприо", "leonardo_dicaprio"),
            ("дикаприо", "leonardo_dicaprio"),
            ("тони старк", "tony_stark"),
            ("железный человек", "tony_stark"),
            ("танос", "thanos"),
            ("мстители", "avengers_general"),
            ("шрек", "shrek"),
            ("губка боб", "spongebob"),
            ("спанч боб", "spongebob"),
            ("патрик", "spongebob_patrick"),
            ("сквидвард", "spongebob_squidward"),
            ("кот", "cat_memes"),
            ("собака", "dog_memes"),
            ("доге", "doge"),
            ("пепе", "pepe_frog"),
            ("человек паук", "spiderman"),
            ("райан гослинг", "ryan_gosling"),
            ("гослинг", "ryan_gosling"),
            ("волк с уолл стрит", "wolf_of_wall_street"),
            ("офис", "the_office"),
            ("майкл скотт", "michael_scott"),
            ("джокер", "joker"),
            ("бэтмен", "batman"),
            ("жириновский", "zhirinovsky"),
        ]

        for phrase, entity_id in known_entities:
            if phrase in combined:
                return entity_id

        # Если явная сущность не совпала, группируем по первому тегу или ключевому слову из user_caption
        if tags:
            clean_tag = re.sub(r"[^\w\s]", "", tags[0]).strip()
            if clean_tag:
                return f"tag_{clean_tag.replace(' ', '_')}"

        if user_caption:
            words = user_caption.split()
            if words:
                clean_word = re.sub(r"[^\w]", "", words[0])
                if clean_word:
                    return f"caption_{clean_word}"

        return f"hash_{record['file_hash'][:8]}"


class DatasetSplitter:
    """Выполняет разбиение пар запросов на train, val и test с изоляцией scene_group."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def build_split(
        self,
        gold_path: Path,
        output_path: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> SplitDataset:
        """Создает и сохраняет фиксированный сплит."""
        with open(gold_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        corpus_hashes = [r["file_hash"] for r in records]

        # Извлекаем все пары (query, corpus_hash, scene_group)
        pairs: list[QueryPair] = []
        for r in records:
            group = SceneGroupDetector.detect_group(r)
            f_hash = r["file_hash"]
            queries = r.get("queries", [])
            for q in queries:
                q_clean = q.strip()
                if q_clean:
                    pairs.append(QueryPair(query=q_clean, file_hash=f_hash, scene_group=group))

        groups = [p.scene_group for p in pairs]
        indices = list(range(len(pairs)))

        # 1-й сплит: Train vs (Val + Test)
        test_val_ratio = 1.0 - train_ratio
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=self._seed)
        train_idx, val_test_idx = next(gss_outer.split(indices, groups=groups))

        # 2-й сплит: Val vs Test
        val_test_groups = [groups[i] for i in val_test_idx]
        val_test_indices = list(range(len(val_test_idx)))
        relative_val_ratio = val_ratio / test_val_ratio

        gss_inner = GroupShuffleSplit(n_splits=1, test_size=1.0 - relative_val_ratio, random_state=self._seed)
        v_idx, t_idx = next(gss_inner.split(val_test_indices, groups=val_test_groups))

        val_idx = [val_test_idx[i] for i in v_idx]
        test_idx = [val_test_idx[i] for i in t_idx]

        train_pairs = [{"query": pairs[i].query, "file_hash": pairs[i].file_hash} for i in train_idx]
        val_pairs = [{"query": pairs[i].query, "file_hash": pairs[i].file_hash} for i in val_idx]
        test_pairs = [{"query": pairs[i].query, "file_hash": pairs[i].file_hash} for i in test_idx]

        split_data = SplitDataset(
            version="v1",
            seed=self._seed,
            corpus_hashes=corpus_hashes,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(split_data), f, ensure_ascii=False, indent=2)

        logger.info(
            "Сплит сохранен в %s: Corpus=%d, Train=%d, Val=%d, Test=%d",
            output_path,
            len(corpus_hashes),
            len(train_pairs),
            len(val_pairs),
            len(test_pairs),
        )
        return split_data


def main() -> None:
    """Точка входа генератора сплитов."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    gold_file = Path("data/gold/gold_dataset.jsonl")
    out_file = Path("data/splits/splits_v1.json")
    splitter = DatasetSplitter(seed=42)
    res = splitter.build_split(gold_file, out_file)
    logger.info("Готово: Train=%d, Val=%d, Test=%d", len(res.train_pairs), len(res.val_pairs), len(res.test_pairs))


if __name__ == "__main__":
    main()
