import html
from pathlib import Path
from typing import ClassVar

from meme_organizer.labeling.store import LabelingCheckpointStore
from meme_organizer.schemas.labeling import MediaType, SilverRecord


class SilverHtmlReviewer:
    """Генерирует локальный автономный HTML-файл для быстрого визуального ревью silver-разметки."""

    DEFAULT_FILENAME: ClassVar[str] = "review.html"

    def __init__(self, silver_dir: Path) -> None:
        self._silver_dir = silver_dir

    def build(self, output_path: Path | None = None) -> Path:
        """Читает labels.jsonl и failed.jsonl и сохраняет review.html."""
        store = LabelingCheckpointStore(self._silver_dir)
        records = store.all_labels()
        failed = store.all_failed()

        target = output_path or (self._silver_dir / self.DEFAULT_FILENAME)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._render_document(records, len(failed)), encoding="utf-8")
        return target

    @classmethod
    def _render_document(cls, records: list[SilverRecord], failed_count: int) -> str:
        cards = "\n".join(cls._render_card(rec) for rec in records)
        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Meme Organizer — Silver Review ({len(records)} items)</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 20px; }}
  .stats {{ margin-bottom: 18px; color: #aaa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
  .card {{ background: #1c1c1e; border: 1px solid #333; border-radius: 10px; padding: 12px; }}
  .media {{ width: 100%; max-height: 260px; object-fit: contain; background: #000; border-radius: 6px; }}
  .slug {{ font-family: monospace; color: #7dd3fc; font-size: 14px; margin: 8px 0 4px; word-break: break-all; }}
  .orig {{ font-size: 11px; color: #888; margin-bottom: 8px; }}
  .caption {{ font-weight: 600; color: #f4f4f5; margin: 6px 0 4px; font-size: 14px; }}
  .rich {{ font-size: 12px; color: #a1a1aa; margin-bottom: 6px; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }}
  .tag {{ background: #2c2c2e; color: #c4b5fd; padding: 2px 8px; border-radius: 12px; font-size: 12px; }}
  .queries {{ margin: 6px 0; padding-left: 18px; font-size: 13px; color: #d1d5db; }}
  .transcript {{ font-size: 12px; color: #fde68a; background: #292524; padding: 6px; border-radius: 6px; margin-top: 6px; }}
</style>
</head>
<body>
  <h1>Silver Dataset Review</h1>
  <div class="stats">Размечено: <b>{len(records)}</b> | Ошибок/блокировок (failed.jsonl): <b>{failed_count}</b></div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""

    @classmethod
    def _render_card(cls, rec: SilverRecord) -> str:
        src = html.escape(rec.silver_filename)
        if rec.media_type == MediaType.VIDEO:
            media_tag = f'<video class="media" src="{src}" controls preload="metadata"></video>'
        else:
            media_tag = f'<img class="media" src="{src}" loading="lazy" alt="{src}">'

        tags_html = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in rec.label.tags)
        queries_html = "".join(f"<li>{html.escape(q)}</li>" for q in rec.label.queries)
        transcript_block = (
            f'<div class="transcript"><b>Текст/речь:</b> {html.escape(rec.label.transcript_or_text)}</div>'
            if rec.label.transcript_or_text.strip()
            else ""
        )

        return f"""<div class="card">
  {media_tag}
  <div class="slug">{src}</div>
  <div class="orig">orig: {html.escape(rec.original_filename)} | {rec.media_type.value}</div>
  <div class="caption">{html.escape(rec.label.user_caption)}</div>
  <div class="rich">{html.escape(rec.label.rich_description)}</div>
  <div class="tags">{tags_html}</div>
  {transcript_block}
  <ul class="queries">{queries_html}</ul>
</div>"""
