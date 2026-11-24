import argparse
import logging
import sys
from pathlib import Path

from meme_organizer.config import LabelingConfig
from meme_organizer.curation.server import CurationAppFactory
from meme_organizer.labeling.gemini_client import GeminiLabeler
from meme_organizer.labeling.pipeline import BronzeLabelingPipeline
from meme_organizer.labeling.rate_limiter import RateLimiter
from meme_organizer.labeling.reviewer import SilverHtmlReviewer
from meme_organizer.labeling.store import LabelingCheckpointStore
from meme_organizer.media.normalizer import BronzeNormalizer
from meme_organizer.media.preprocessor import MediaPreprocessor
from meme_organizer.media.scanner import MediaScanner
from meme_organizer.media.slugger import SlugManager
from meme_organizer.utils.seed import SeedManager

logger = logging.getLogger(__name__)


class LabelingCli:
    """Точка входа CLI для запуска авторазметки bronze -> silver и генерации HTML-ревью."""

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Создаёт парсер аргументов CLI."""
        parser = argparse.ArgumentParser(prog="meme-label", description="Meme Organizer dataset labeling CLI")
        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            help="Уровень логирования (по умолчанию INFO)",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        label_cmd = subparsers.add_parser("label", help="Разметить data/bronze -> data/silver через Gemini API")
        label_cmd.add_argument("--bronze-dir", type=Path, default=None, help="Путь к исходной папке bronze")
        label_cmd.add_argument("--silver-dir", type=Path, default=None, help="Путь к целевой папке silver")
        label_cmd.add_argument("--model", type=str, default=None, help="Имя модели Gemini (по умолчанию из .env)")
        label_cmd.add_argument("--rpm", type=float, default=None, help="Лимит запросов в минуту (RPM)")
        label_cmd.add_argument("--max-retries", type=int, default=None, help="Число повторов при 503/429")
        label_cmd.add_argument("--max-files", type=int, default=None, help="Максимум файлов за один запуск")
        label_cmd.add_argument("--retry-failed", action="store_true", help="Повторить только файлы из failed.jsonl")
        label_cmd.add_argument("--rename-in-place", action="store_true", help="Переименовывать файлы прямо в bronze")
        label_cmd.add_argument("--seed", type=int, default=None, help="Random seed")

        review_cmd = subparsers.add_parser("review", help="Сгенерировать data/silver/review.html")
        review_cmd.add_argument("--silver-dir", type=Path, default=None, help="Путь к папке silver")

        norm_cmd = subparsers.add_parser(
            "normalize",
            help="Конвертировать замаскированные HEIF/WEBP в data/bronze в честный .jpg",
        )
        norm_cmd.add_argument("--bronze-dir", type=Path, default=None, help="Путь к исходной папке bronze")

        curate_cmd = subparsers.add_parser("curate", help="Запустить локальный UI для доразметки и слияния дублей")
        curate_cmd.add_argument("--port", type=int, default=8080, help="Порт веб-сервера")
        curate_cmd.add_argument("--host", type=str, default="127.0.0.1", help="Хост веб-сервера")
        curate_cmd.add_argument("--silver-dir", type=Path, default=None, help="Путь к silver папке")
        curate_cmd.add_argument("--gold-dir", type=Path, default=None, help="Путь к gold папке")

        return parser

    @classmethod
    def run(cls, argv: list[str] | None = None) -> int:
        """Разбирает аргументы и выполняет выбранную подкоманду."""
        args = cls.build_parser().parse_args(argv)
        cls._configure_logging(args.log_level)
        config = LabelingConfig()

        if args.command == "review":
            silver_dir = args.silver_dir or config.silver_dir
            output = SilverHtmlReviewer(silver_dir).build()
            logger.info("HTML-ревьюер сохранён: %s", output)
            return 0

        if args.command == "normalize":
            bronze_dir = args.bronze_dir or config.bronze_dir
            res = BronzeNormalizer().normalize_dir(bronze_dir)
            logger.info(
                "Нормализация завершена: проверено=%d, конвертировано=%d, ошибок=%d",
                res.scanned_images,
                res.converted_count,
                res.errors_count,
            )
            return 1 if res.errors_count else 0

        if args.command == "curate":
            import uvicorn

            silver_dir = args.silver_dir or config.silver_dir
            gold_dir = args.gold_dir or config.gold_dir
            app = CurationAppFactory.create_app(silver_dir, gold_dir)
            logger.info("Запуск UI доразметки на http://%s:%d", args.host, args.port)
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
            return 0

        return cls._run_label(args, config)

    @staticmethod
    def _configure_logging(level_name: str) -> None:
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        if level > logging.DEBUG:
            for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "google_genai"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

    @staticmethod
    def _run_label(args: argparse.Namespace, config: LabelingConfig) -> int:
        seed = args.seed if args.seed is not None else config.random_seed
        SeedManager.seed_everything(seed)

        bronze_dir = args.bronze_dir or config.bronze_dir
        silver_dir = args.silver_dir or config.silver_dir
        model_name = args.model or config.model_name
        rpm = args.rpm if args.rpm is not None else config.rpm
        max_retries = args.max_retries if args.max_retries is not None else config.max_retries
        rename_in_place = bool(args.rename_in_place or config.rename_in_place)

        pipeline = BronzeLabelingPipeline(
            scanner=MediaScanner(),
            preprocessor=MediaPreprocessor(
                image_max_side=config.image_max_side,
                image_max_mb=config.image_max_mb,
                video_max_mb=config.video_max_mb,
                video_target_height=config.video_target_height,
            ),
            labeler=GeminiLabeler(
                api_key=config.gemini_api_key.get_secret_value(),
                model_name=model_name,
                inline_max_mb=config.inline_max_mb,
            ),
            store=LabelingCheckpointStore(silver_dir),
            slugger=SlugManager(silver_dir, rename_in_place=rename_in_place),
            rate_limiter=RateLimiter(
                rpm=rpm,
                max_retries=max_retries,
                base_backoff_sec=config.base_backoff_sec,
            ),
        )

        summary = pipeline.run(
            bronze_dir,
            max_files=args.max_files,
            retry_failed=args.retry_failed,
        )
        logger.info(
            "Итог: найдено=%d, пропущено=%d, размечено=%d, ошибок=%d, стоп_по_квоте=%s",
            summary.total_discovered,
            summary.skipped_already_labeled,
            summary.processed_success,
            summary.failed_count,
            summary.stopped_by_quota,
        )
        return 2 if summary.stopped_by_quota else 0


def main() -> None:
    """Точка входа консольного скрипта meme-label."""
    sys.exit(LabelingCli.run())


if __name__ == "__main__":
    main()
