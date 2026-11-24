import importlib
import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)


class SeedManager:
    """Утилита детерминированного сидирования для стандартного random и доступных ML-библиотек."""

    DEFAULT_SEED: int = 42

    @classmethod
    def seed_everything(cls, seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> list[str]:
        """Фиксирует сиды в Python, NumPy и PyTorch (если соответствующие библиотеки установлены)."""
        seeded: list[str] = []

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        seeded.append("python")

        if cls._seed_numpy(seed):
            seeded.append("numpy")

        if cls._seed_torch(seed, deterministic=deterministic_torch):
            seeded.append("torch")

        logger.debug("Seeded libraries (%s) with seed=%d", ", ".join(seeded), seed)
        return seeded

    @staticmethod
    def _try_import(module_name: str) -> Any | None:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return None

    @classmethod
    def _seed_numpy(cls, seed: int) -> bool:
        np = cls._try_import("numpy")
        if np is None:
            return False
        np.random.seed(seed)
        return True

    @classmethod
    def _seed_torch(cls, seed: int, *, deterministic: bool) -> bool:
        torch = cls._try_import("torch")
        if torch is None:
            return False

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        return True
