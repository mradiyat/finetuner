"""Dataset preparation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class CaptionRecord:
    """A single image/caption pair."""

    image_path: Path
    caption: Optional[str]


def read_metadata_jsonl(path: Path) -> List[CaptionRecord]:
    """Load caption records from a metadata.jsonl file."""

    records: List[CaptionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image_path = path.parent / payload["file_name"]
            caption = payload.get("text")
            records.append(CaptionRecord(image_path=image_path, caption=caption))
    return records


def iter_dataset_files(dataset_root: Path) -> Iterable[CaptionRecord]:
    """Iterate through image/caption pairs under ``dataset_root``."""

    metadata_path = dataset_root / "metadata.jsonl"
    if metadata_path.exists():
        yield from read_metadata_jsonl(metadata_path)
        return

    data_dir = dataset_root / "data"
    search_root = data_dir if data_dir.exists() else dataset_root

    for image_path in sorted(search_root.rglob("*")):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        caption_path = image_path.with_suffix(".txt")
        caption = None
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
        yield CaptionRecord(image_path=image_path, caption=caption)


def validate_images(records: Iterable[CaptionRecord]) -> List[str]:
    """Validate dataset records and return a list of warnings."""

    warnings: List[str] = []
    for record in records:
        if not record.image_path.exists():
            warnings.append(f"Missing image file: {record.image_path}")
            continue
        try:
            with Image.open(record.image_path) as img:
                img.verify()
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"Failed to load {record.image_path}: {exc}")
        if record.caption is not None and not record.caption:
            warnings.append(f"Empty caption for {record.image_path}")
    return warnings
