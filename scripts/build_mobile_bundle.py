"""CLI-скрипт для сборки и экспорта мобильного пакета поставки."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meme_organizer.export.bundle.mobile_bundle_builder import MobileBundleBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_mobile_bundle")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка Mobile Delivery Bundle для Android приложения.")
    parser.add_argument(
        "--profile",
        type=str,
        default="text-champion",
        help="ID профиля модели для сборки (по умолчанию: text-champion). Варианты: text-champion, rubert-tiny2-finetuned",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/mobile_bundle"),
        help="Целевая директория для сохранения готового архива (по умолчанию: dist/mobile_bundle)",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default=None,
        help="Тег релиза для записи в preset_manifest.json (например: v0.2.0-mobile)",
    )
    args = parser.parse_args()

    try:
        profile = MobileBundleBuilder.get_profile(args.profile)
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)

    work_dir = Path(".cache/bundle_build")
    builder = MobileBundleBuilder(profile=profile, work_dir=work_dir)

    target_archive = args.output_dir / f"meme-organizer-{profile.profile_id}.tar.gz"
    output_path = builder.build(output_tar=target_archive, release_tag=args.release_tag)

    logger.info("Успешно завершено! Архив: %s", output_path.resolve())


if __name__ == "__main__":
    main()
