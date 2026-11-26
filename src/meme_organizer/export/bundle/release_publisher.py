"""Публикация локально собранных мобильных бандлов в GitHub Releases."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class GitHubReleasePublisher:
    """Загружает мобильные бандлы и контрольные суммы в GitHub Releases через REST API."""

    API_BASE = "https://api.github.com"

    def __init__(self, token: str, owner: str | None = None, repo: str | None = None) -> None:
        if not token.strip():
            raise ValueError("GITHUB_TOKEN не задан. Укажите токен через --token, переменную окружения или .env.")
        self.token = token.strip()
        if owner and repo:
            self.owner, self.repo = owner, repo
        else:
            self.owner, self.repo = self.get_repo_from_git()

    @staticmethod
    def parse_github_repo(remote_url: str) -> tuple[str, str]:
        """Извлекает пару (owner, repo) из SSH или HTTPS URL удаленного репозитория GitHub; при неверном формате вызывает ValueError."""
        url = remote_url.strip()
        match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
        if not match:
            raise ValueError(f"Не удалось определить owner/repo из remote URL: {url}")
        return match.group(1), match.group(2)

    @classmethod
    def get_repo_from_git(cls) -> tuple[str, str]:
        """Читает URL удаленного репозитория origin из локального git и возвращает (owner, repo); при отсутствии git вызывает RuntimeError."""
        git_bin = shutil.which("git")
        if not git_bin:
            raise RuntimeError("Утилита git не найдена в PATH.")
        out = subprocess.check_output(  # noqa: S603
            [git_bin, "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
        )
        return cls.parse_github_repo(out.decode("utf-8"))

    @staticmethod
    def write_sha256sums(archive_path: Path, output_path: Path | None = None) -> Path:
        """Вычисляет SHA-256 файла archive_path, записывает файл контрольных сумм и возвращает путь к нему."""
        h = hashlib.sha256()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        digest = h.hexdigest()

        sums_path = output_path or (archive_path.parent / "SHA256SUMS.txt")
        sums_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
        return sums_path

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def publish(
        self,
        tag: str,
        files: list[Path],
        title: str | None = None,
        body: str | None = None,
    ) -> str:
        """Создает или обновляет релиз GitHub для тега tag, загружает файлы files и возвращает URL релиза; при отсутствии файла вызывает FileNotFoundError."""
        release_title = title or f"Mobile Delivery Bundle: {tag}"
        release_body = body or self._default_release_body(files)

        with requests.Session() as session:
            session.headers.update(self._headers())
            release = self._get_or_create_release(session, tag, release_title, release_body)
            upload_url = release["upload_url"].split("{")[0]
            existing_assets = {a["name"]: a["id"] for a in release.get("assets", [])}

            for file_path in files:
                if not file_path.exists():
                    raise FileNotFoundError(f"Файл для загрузки не найден: {file_path}")

                if file_path.name in existing_assets:
                    asset_id = existing_assets[file_path.name]
                    logger.info("Удаление существующего ассета %s (id=%s)...", file_path.name, asset_id)
                    del_resp = session.delete(
                        f"{self.API_BASE}/repos/{self.owner}/{self.repo}/releases/assets/{asset_id}",
                        timeout=30,
                    )
                    del_resp.raise_for_status()

                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info("Загрузка %s (%.2f МБ) в релиз %s...", file_path.name, size_mb, tag)
                with open(file_path, "rb") as f:
                    up_resp = session.post(
                        upload_url,
                        params={"name": file_path.name},
                        headers={"Content-Type": "application/octet-stream"},
                        data=f,
                        timeout=300,
                    )
                up_resp.raise_for_status()
                logger.info("Загружен ассет: %s", up_resp.json().get("browser_download_url"))

            return str(release["html_url"])

    def _get_or_create_release(
        self,
        session: requests.Session,
        tag: str,
        title: str,
        body: str,
    ) -> dict[str, Any]:
        url = f"{self.API_BASE}/repos/{self.owner}/{self.repo}/releases/tags/{tag}"
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            logger.info("Найден существующий релиз для тега %s.", tag)
            return dict(resp.json())

        if resp.status_code != 404:
            resp.raise_for_status()

        logger.info("Создание нового релиза %s в %s/%s...", tag, self.owner, self.repo)
        create_resp = session.post(
            f"{self.API_BASE}/repos/{self.owner}/{self.repo}/releases",
            json={
                "tag_name": tag,
                "name": title,
                "body": body,
                "draft": False,
                "prerelease": False,
            },
            timeout=30,
        )
        create_resp.raise_for_status()
        return dict(create_resp.json())

    @staticmethod
    def _default_release_body(files: list[Path]) -> str:
        file_list = "\n".join(f"- `{p.name}`" for p in files)
        return f"Meme Organizer Mobile Text Engine (ONNX Dynamic INT8).\n\nФайлы релиза:\n{file_list}\n"
