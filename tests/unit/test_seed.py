import os
import random

from meme_organizer.utils.seed import SeedManager


def test_seed_everything_reproducibility() -> None:
    seeded = SeedManager.seed_everything(123)
    assert "python" in seeded
    assert os.environ["PYTHONHASHSEED"] == "123"
    seq1 = [random.random() for _ in range(5)]

    SeedManager.seed_everything(123)
    seq2 = [random.random() for _ in range(5)]
    assert seq1 == seq2
