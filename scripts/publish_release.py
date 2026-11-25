"""CLI script for locally building a Mobile Delivery Bundle and publishing it to GitHub Releases."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from meme_organizer.mobile.bundle_builder import MobileBundleBuilder, get_profile
from meme_organizer.mobile.release_publisher import GitHubReleasePublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("publish_release")


class ReleaseEnv(BaseSettings):
    """Loads GITHUB_TOKEN from environment or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    github_token: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Локальная сборка Mobile Delivery Bundle из MLflow и публикация в GitHub Releases."
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Тег релиза на GitHub (например: v0.2.0-mobile)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="text-champion",
        help="Профиль модели (по умолчанию: text-champion)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/mobile_bundle"),
        help="Директория сборки архива (по умолчанию: dist/mobile_bundle)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Не пересобирать архив, а загрузить уже существующий из --output-dir",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub Personal Access Token (по умолчанию берется из GITHUB_TOKEN в окружении или .env)",
    )
    args = parser.parse_args()

    try:
        profile = get_profile(args.profile)
    except KeyError as exc:
        logger.error(str(exc))
        sys.exit(1)

    token = args.token or os.environ.get("GITHUB_TOKEN") or ReleaseEnv().github_token
    if not token:
        logger.error("Не задан GITHUB_TOKEN. Передайте --token или добавьте GITHUB_TOKEN в .env / окружение.")
        sys.exit(1)

    target_archive = args.output_dir / f"meme-organizer-{profile.profile_id}.tar.gz"

    if not args.skip_build:
        builder = MobileBundleBuilder(profile=profile, work_dir=Path(".cache/bundle_build"))
        target_archive = builder.build(output_tar=target_archive)
    elif not target_archive.exists():
        logger.error("Архив не найден (--skip-build): %s", target_archive)
        sys.exit(1)

    sums_path = GitHubReleasePublisher.write_sha256sums(target_archive)
    publisher = GitHubReleasePublisher(token=token)
    release_url = publisher.publish(tag=args.tag, files=[target_archive, sums_path])

    logger.info("Релиз опубликован: %s", release_url)


if __name__ == "__main__":
    main()
