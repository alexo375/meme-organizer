from pathlib import Path

from PIL import Image

from meme_organizer.media.normalizer import BronzeNormalizer


def test_normalizer_converts_mismatched_webp_to_real_jpeg(tmp_path: Path) -> None:
    disguised = tmp_path / "fake.jpg"
    valid_png = tmp_path / "real.png"

    Image.new("RGB", (32, 32), color=(120, 40, 200)).save(disguised, format="WEBP")
    Image.new("RGB", (32, 32), color=(10, 200, 10)).save(valid_png, format="PNG")

    result = BronzeNormalizer().normalize_dir(tmp_path)

    assert result.scanned_images == 2
    assert result.converted_count == 1
    assert result.errors_count == 0
    assert disguised.exists()
    with Image.open(disguised) as img:
        assert img.format == "JPEG"
    with Image.open(valid_png) as img:
        assert img.format == "PNG"
