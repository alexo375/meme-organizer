"""Leak-free dataset splitter grouping related memes by scene_group."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from meme_organizer.dataset.schemas import QueryPair

logger = logging.getLogger(__name__)


class LeakFreeSplitter:
    """Splits query pairs into Train / Val / Test using GroupShuffleSplit on scene_group."""

    def __init__(
        self,
        gold_dataset_path: Path,
        output_splits_path: Path = Path("data/splits/splits_v1.json"),
        random_seed: int = 42,
    ) -> None:
        self.gold_dataset_path = gold_dataset_path
        self.output_splits_path = output_splits_path
        self.random_seed = random_seed

    def generate_splits(
        self,
        train_ratio: float = 0.75,
        val_ratio: float = 0.125,
        test_ratio: float = 0.125,
    ) -> dict[str, Any]:
        """Split query pairs while keeping related scene groups in the same split."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-5:
            raise ValueError("Сумма train_ratio, val_ratio и test_ratio должна быть равна 1.0")

        with open(self.gold_dataset_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        corpus_hashes = [r["file_hash"] for r in records]

        query_pairs: list[QueryPair] = []
        for r in records:
            f_hash = r["file_hash"]
            scene_group = r.get("extra", {}).get("scene_group") or f_hash
            for q in r.get("queries", []):
                q_clean = q.strip()
                if q_clean:
                    query_pairs.append(
                        QueryPair(
                            query=q_clean,
                            file_hash=f_hash,
                            scene_group=scene_group,
                        )
                    )

        groups = np.array([p.scene_group for p in query_pairs])
        indices = np.arange(len(query_pairs))

        # Split 1: Train vs (Val + Test)
        test_val_ratio = val_ratio + test_ratio
        gss_main = GroupShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=self.random_seed)
        train_idx, val_test_idx = next(gss_main.split(indices, groups=groups))

        # Split 2: Val vs Test
        val_test_groups = groups[val_test_idx]
        val_test_sub_indices = np.arange(len(val_test_idx))
        val_rel_ratio = val_ratio / test_val_ratio

        gss_sub = GroupShuffleSplit(n_splits=1, test_size=1.0 - val_rel_ratio, random_state=self.random_seed)
        sub_val_idx, sub_test_idx = next(gss_sub.split(val_test_sub_indices, groups=val_test_groups))

        val_idx = val_test_idx[sub_val_idx]
        test_idx = val_test_idx[sub_test_idx]

        split_dict: dict[str, Any] = {
            "version": "v1",
            "random_seed": self.random_seed,
            "corpus_hashes": corpus_hashes,
            "train_pairs": [query_pairs[i].model_dump() for i in train_idx],
            "val_pairs": [query_pairs[i].model_dump() for i in val_idx],
            "test_pairs": [query_pairs[i].model_dump() for i in test_idx],
        }

        self.output_splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_splits_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2, ensure_ascii=False)

        logger.info(
            "Сгенерирован сплит: %d train, %d val, %d test",
            len(split_dict["train_pairs"]),
            len(split_dict["val_pairs"]),
            len(split_dict["test_pairs"]),
        )
        return split_dict
