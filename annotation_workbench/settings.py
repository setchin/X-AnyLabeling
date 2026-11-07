"""Load optional machine-local paths without putting them in the repository."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Paths for the review store and independently installed inference models."""

    data_root: Path
    output_root: Path
    raw_predictions: Path | None = None
    anchor_references: Path | None = None
    anchor_alignment: Path | None = None
    coco_annotations: Path | None = None
    sam3_repo: Path | None = None
    sam3_checkpoint: Path | None = None
    deim_repo: Path | None = None
    deim_config: Path | None = None
    deim_checkpoint: Path | None = None
    gui_site: Path | None = None


def load_settings(filename: str | Path | None = None) -> Settings:
    """Resolve configured paths relative to the JSON file.

    Args:
        filename: Configuration file; defaults to FOREST_SAR_CONFIG, if set.

    Returns:
        Settings with absolute paths. No configuration is needed for unit tests.
    """
    filename = filename or os.environ.get('FOREST_SAR_CONFIG')
    values = {}
    base = ROOT
    if filename:
        path = Path(filename).expanduser().resolve()
        values = json.loads(path.read_text())
        base = path.parent
    unknown = set(values) - set(Settings.__dataclass_fields__)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    paths = {name: (base / Path(value).expanduser()).resolve()
             for name, value in values.items() if value is not None}
    paths.setdefault('data_root', ROOT / 'data')
    paths.setdefault('output_root', ROOT / 'review_output')
    return Settings(**paths)


SETTINGS = load_settings()
