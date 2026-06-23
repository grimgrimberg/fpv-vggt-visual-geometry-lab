from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from .schemas import CatalogRecord, CatalogSnapshot, model_to_dict


DEFAULT_README_URL = (
    "https://raw.githubusercontent.com/itamarwe/"
    "fpv-drone-strikes-lebanon-dataset/main/README.md"
)
DEFAULT_MANIFEST_URL = (
    "https://raw.githubusercontent.com/itamarwe/"
    "fpv-drone-strikes-lebanon-dataset/main/"
    "2026-06-21_fpv_renamed_from_first_frames_manifest.tsv"
)


def sync_catalog(
    readme_source: str,
    output: Path,
    manifest_source: str | None = None,
) -> tuple[Path, Path]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    readme_text = read_text_source(readme_source)
    manifest_text = read_text_source(manifest_source) if manifest_source else None

    records = parse_readme_catalog(readme_text)
    manifest_rows = parse_manifest_tsv(manifest_text) if manifest_text else {}
    merged = [merge_manifest(record, manifest_rows) for record in records]

    catalog_path = output / "catalog.parquet"
    pd.DataFrame([model_to_dict(record) for record in merged]).to_parquet(
        catalog_path, index=False
    )

    snapshot = CatalogSnapshot(
        readme_source=readme_source,
        manifest_source=manifest_source,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        row_count=len(merged),
        readme_sha256=sha256_text(readme_text),
        manifest_sha256=sha256_text(manifest_text) if manifest_text is not None else None,
    )
    snapshot_path = output / "catalog_snapshot.json"
    snapshot_path.write_text(
        json.dumps(model_to_dict(snapshot), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return catalog_path, snapshot_path


def parse_readme_catalog(readme_text: str) -> list[CatalogRecord]:
    records: list[CatalogRecord] = []
    for line in readme_text.splitlines():
        if not re.match(r"^\|\s*20\d\d-\d\d-\d\d\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        date, image_cell, description, link_cell = cells[:4]
        video_url = _extract_markdown_url(link_cell)
        if not video_url:
            continue
        thumbnail_url = _extract_img_src(image_cell)
        video_id = Path(urlparse(video_url).path).stem
        records.append(
            CatalogRecord(
                video_id=video_id,
                date=date,
                source_description=description,
                video_url=video_url,
                thumbnail_url=thumbnail_url,
            )
        )
    return records


def parse_manifest_tsv(manifest_text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(manifest_text.splitlines(), delimiter="\t")
    for row in reader:
        current = row.get("current_stem", "")
        target = row.get("target_stem", "")
        if current:
            rows[current] = row
        if target:
            rows[target] = row
    return rows


def merge_manifest(
    record: CatalogRecord, manifest_rows: dict[str, dict[str, str]]
) -> CatalogRecord:
    row = manifest_rows.get(record.video_id)
    if not row:
        return record
    return record.model_copy(
        update={
            "current_stem": row.get("current_stem") or None,
            "target_stem": row.get("target_stem") or None,
            "manifest_confidence": row.get("confidence") or None,
            "manifest_notes": row.get("notes") or None,
        }
    )


def read_catalog(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def shortlist_catalog(catalog_path: Path, limit: int = 10) -> pd.DataFrame:
    catalog = read_catalog(catalog_path)
    columns = ["video_id", "date", "source_description", "video_url"]
    return catalog.loc[:, [column for column in columns if column in catalog.columns]].head(limit)


def read_text_source(source: str | Path | None) -> str:
    if source is None:
        raise ValueError("source is required")
    source_text = str(source)
    parsed = urlparse(source_text)
    if parsed.scheme in {"http", "https", "file"}:
        with urlopen(source_text) as response:
            return response.read().decode("utf-8")
    return Path(source_text).read_text(encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_markdown_url(cell: str) -> str | None:
    match = re.search(r"\[[^\]]+\]\(([^)]+)\)", cell)
    return match.group(1) if match else None


def _extract_img_src(cell: str) -> str | None:
    match = re.search(r"src=[\"']([^\"']+)[\"']", cell)
    return match.group(1) if match else None
