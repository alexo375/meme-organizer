from pathlib import Path

from PIL import Image

from meme_organizer.labeling.exceptions import QuotaExhaustedError, SafetyBlockedError
from meme_organizer.labeling.pipeline import BronzeLabelingPipeline
from meme_organizer.labeling.rate_limiter import RateLimiter
from meme_organizer.labeling.reviewer import SilverHtmlReviewer
from meme_organizer.labeling.store import LabelingCheckpointStore
from meme_organizer.media.preprocessor import MediaPreprocessor, PreparedMedia
from meme_organizer.media.scanner import MediaScanner
from meme_organizer.media.slugger import SlugManager
from meme_organizer.schemas.labeling import MediaFileInfo, MemeLabelResponse


class FakeLabeler:
    """Фейковый разметчик для модульного тестирования пайплайна без сети."""

    def __init__(self, fail_map: dict[str, Exception] | None = None) -> None:
        self._fail_map = fail_map or {}
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-gemini-v1"

    def label(self, info: MediaFileInfo, prepared: PreparedMedia) -> MemeLabelResponse:
        self.calls.append(info.original_path.name)
        if info.original_path.name in self._fail_map:
            raise self._fail_map[info.original_path.name]
        stem = info.original_path.stem
        return MemeLabelResponse(
            slug=f"meme-{stem}",
            user_caption=f"подпись {stem}",
            rich_description=f"Подробное описание для {stem}",
            tags=["мем", "реакция", stem],
            transcript_or_text="привет",
            queries=[f"запрос {stem}", "реакция", "цитата", "для переговоров"],
        )


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path, format="JPEG")


def test_pipeline_resume_safety_and_review_html(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    bronze.mkdir()

    _create_image(bronze / "a.jpg", (255, 0, 0))
    _create_image(bronze / "b.jpg", (0, 255, 0))
    _create_image(bronze / "c.jpg", (0, 0, 255))

    labeler = FakeLabeler(fail_map={"b.jpg": SafetyBlockedError("nsfw")})
    store = LabelingCheckpointStore(silver)
    pipeline = BronzeLabelingPipeline(
        scanner=MediaScanner(ffprobe_bin=None),
        preprocessor=MediaPreprocessor(ffmpeg_bin=None),
        labeler=labeler,
        store=store,
        slugger=SlugManager(silver, rename_in_place=False),
        rate_limiter=RateLimiter(rpm=6000.0, sleep_fn=lambda _: None),
    )

    first_run = pipeline.run(bronze, max_files=2, show_progress=False)
    assert first_run.processed_success == 1
    assert first_run.failed_count == 1
    assert (silver / "meme-a.jpg").exists()

    second_run = pipeline.run(bronze, show_progress=False)
    assert second_run.skipped_already_labeled == 1
    assert second_run.processed_success == 1
    assert (silver / "meme-c.jpg").exists()
    assert len(store.all_labels()) == 2
    assert len(store.all_failed()) == 1

    review_path = SilverHtmlReviewer(silver).build()
    html_text = review_path.read_text(encoding="utf-8")
    assert "meme-a.jpg" in html_text
    assert "meme-c.jpg" in html_text


def test_pipeline_stops_cleanly_on_quota_exhausted(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    bronze.mkdir()

    _create_image(bronze / "1.jpg", (10, 20, 30))
    _create_image(bronze / "2.jpg", (40, 50, 60))

    labeler = FakeLabeler(fail_map={"1.jpg": QuotaExhaustedError("RPD limit")})
    pipeline = BronzeLabelingPipeline(
        scanner=MediaScanner(ffprobe_bin=None),
        preprocessor=MediaPreprocessor(ffmpeg_bin=None),
        labeler=labeler,
        store=LabelingCheckpointStore(silver),
        slugger=SlugManager(silver, rename_in_place=False),
        rate_limiter=RateLimiter(rpm=6000.0, sleep_fn=lambda _: None),
    )

    summary = pipeline.run(bronze, show_progress=False)
    assert summary.stopped_by_quota is True
    assert labeler.calls == ["1.jpg"]
