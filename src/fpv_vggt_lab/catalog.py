from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
DEFAULT_BANNER_AUDIT_URL = (
    "https://raw.githubusercontent.com/itamarwe/"
    "fpv-drone-strikes-lebanon-dataset/main/2026-07-05_banner_title_audit.tsv"
)
DEFAULT_GEO_RECORDS_URL = (
    "https://raw.githubusercontent.com/itamarwe/"
    "fpv-drone-strikes-lebanon-dataset/main/geo/fpv_drone_map_records.csv"
)
DEFAULT_WEBSITE_URL = "https://www.itamarweiss.com/fpv/"
CLOUDFRONT_VIDEO_BASE_URL = "https://d2fioemadmrru3.cloudfront.net/videos"
CLOUDFRONT_THUMBNAIL_BASE_URL = "https://d2fioemadmrru3.cloudfront.net/thumbnails"
SOURCE_METADATA_WARNING = (
    "source metadata only; do not use for geolocation, map projection, route, "
    "approach, launch, target-coordinate, speed, standoff, guidance, or maneuver inference"
)
FALLBACK_CATALOG_WARNING = (
    "catalog reconciled with previous local catalog rows because the latest upstream "
    "metadata exposes fewer rows than the declared MP4 storage count"
)


def sync_catalog(
    readme_source: str,
    output: Path,
    manifest_source: str | None = None,
    banner_audit_source: str | None = None,
    geo_records_source: str | None = None,
    website_source: str | None = None,
    fallback_catalog_source: str | Path | None = None,
) -> tuple[Path, Path]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    readme_text = read_text_source(readme_source)
    manifest_text = read_text_source(manifest_source) if manifest_source else None
    banner_text = read_text_source(banner_audit_source) if banner_audit_source else None
    geo_text = read_text_source(geo_records_source) if geo_records_source else None
    website_text = read_text_source(website_source) if website_source else None

    readme_records = parse_readme_catalog(readme_text)
    manifest_rows = parse_manifest_tsv(manifest_text) if manifest_text else {}
    banner_rows = parse_banner_title_audit_tsv(banner_text) if banner_text else {}
    geo_rows = parse_geo_records_csv(geo_text) if geo_text else {}
    declared_count = parse_declared_video_count(readme_text)
    records = build_catalog_records(
        readme_records=readme_records,
        manifest_rows=manifest_rows,
        geo_rows=geo_rows,
        declared_count=declared_count,
        fallback_catalog_source=fallback_catalog_source,
    )
    merged = [merge_source_metadata(record, manifest_rows, banner_rows, geo_rows) for record in records]

    catalog_path = output / "catalog.parquet"
    pd.DataFrame([model_to_dict(record) for record in merged]).to_parquet(
        catalog_path, index=False
    )

    source_table_paths = write_source_tables(output, banner_rows, geo_rows)
    warnings = [SOURCE_METADATA_WARNING] if banner_rows or geo_rows else []
    if any(record.manifest_confidence == "fallback_previous_catalog" for record in merged):
        warnings.append(FALLBACK_CATALOG_WARNING)
    snapshot = CatalogSnapshot(
        readme_source=readme_source,
        manifest_source=manifest_source,
        banner_audit_source=banner_audit_source,
        geo_records_source=geo_records_source,
        website_source=website_source,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        row_count=len(merged),
        readme_table_row_count=len(readme_records),
        readme_video_url_count=count_readme_video_urls(readme_text),
        upstream_declared_video_count=declared_count,
        readme_sha256=sha256_text(readme_text),
        manifest_sha256=sha256_text(manifest_text) if manifest_text is not None else None,
        banner_audit_sha256=sha256_text(banner_text) if banner_text is not None else None,
        geo_records_sha256=sha256_text(geo_text) if geo_text is not None else None,
        website_sha256=sha256_text(website_text) if website_text is not None else None,
        source_table_paths=source_table_paths,
        source_metadata_warnings=warnings,
    )
    snapshot_path = output / "catalog_snapshot.json"
    snapshot_path.write_text(
        json.dumps(model_to_dict(snapshot), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return catalog_path, snapshot_path


def build_catalog_records(
    readme_records: list[CatalogRecord],
    manifest_rows: dict[str, dict[str, str]],
    geo_rows: dict[str, dict[str, str]],
    declared_count: int | None = None,
    fallback_catalog_source: str | Path | None = None,
) -> list[CatalogRecord]:
    records = readme_records or catalog_records_from_geo_rows(geo_rows)
    records = dedupe_catalog_records(records)
    records = add_manifest_only_records(records, manifest_rows)
    if fallback_catalog_source and declared_count and len(records) < declared_count:
        records = add_fallback_catalog_records(records, fallback_catalog_source, declared_count)
    return records


def catalog_records_from_geo_rows(geo_rows: dict[str, dict[str, str]]) -> list[CatalogRecord]:
    records: list[CatalogRecord] = []
    for video_id, row in geo_rows.items():
        records.append(
            CatalogRecord(
                video_id=video_id,
                date=clean_cell(row.get("date")) or date_from_video_id(video_id),
                source_description=clean_cell(row.get("description")) or description_from_video_id(video_id),
                video_url=clean_cell(row.get("video_url")) or synthesized_video_url(video_id),
                thumbnail_url=clean_cell(row.get("thumbnail_url")) or synthesized_thumbnail_url(video_id),
                source_metadata_warning=SOURCE_METADATA_WARNING,
            )
        )
    return records


def add_manifest_only_records(
    records: list[CatalogRecord],
    manifest_rows: dict[str, dict[str, str]],
) -> list[CatalogRecord]:
    output = list(records)
    existing_ids = {record.video_id for record in output}
    existing_normalized = {normalize_storage_video_id(record.video_id) for record in output}
    seen_rows: set[tuple[str, str]] = set()
    for row in manifest_rows.values():
        current = clean_cell(row.get("current_stem"))
        target = clean_cell(row.get("target_stem"))
        row_key = (current, target)
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        video_id = target or current
        if not video_id:
            continue
        normalized = normalize_storage_video_id(video_id)
        if video_id in existing_ids or normalized in existing_normalized:
            continue
        output.append(
            CatalogRecord(
                video_id=video_id,
                date=clean_cell(row.get("date")) or date_from_video_id(video_id),
                source_description=description_from_video_id(video_id),
                video_url=synthesized_video_url(video_id),
                thumbnail_url=synthesized_thumbnail_url(video_id),
                current_stem=current or None,
                target_stem=target or None,
                manifest_confidence=clean_cell(row.get("confidence")) or None,
                manifest_notes=clean_cell(row.get("notes")) or None,
                source_metadata_warning=SOURCE_METADATA_WARNING,
            )
        )
        existing_ids.add(video_id)
        existing_normalized.add(normalized)
    return output


def add_fallback_catalog_records(
    records: list[CatalogRecord],
    fallback_catalog_source: str | Path,
    declared_count: int,
) -> list[CatalogRecord]:
    output = list(records)
    existing_ids = {record.video_id for record in output}
    existing_normalized = {normalize_storage_video_id(record.video_id) for record in output}
    for fallback in read_fallback_catalog_records(fallback_catalog_source):
        if len(output) >= declared_count:
            break
        normalized = normalize_storage_video_id(fallback.video_id)
        if fallback.video_id in existing_ids or normalized in existing_normalized:
            continue
        old_notes = clean_cell(fallback.manifest_notes)
        if fallback.manifest_confidence:
            old_notes = append_note(old_notes, f"previous confidence={fallback.manifest_confidence}")
        record = fallback.model_copy(
            update={
                "manifest_confidence": "fallback_previous_catalog",
                "manifest_notes": append_note(
                    old_notes,
                    "fallback from previous local catalog; latest upstream declares this storage row count but current public metadata has no replacement row",
                ),
                "source_metadata_warning": SOURCE_METADATA_WARNING,
            }
        )
        output.append(record)
        existing_ids.add(record.video_id)
        existing_normalized.add(normalized)
    return output


def read_fallback_catalog_records(source: str | Path) -> list[CatalogRecord]:
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict("records")
    elif suffix in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])
    elif suffix == ".csv":
        rows = pd.read_csv(path).to_dict("records")
    else:
        raise ValueError(f"unsupported fallback catalog format: {path}")

    records: list[CatalogRecord] = []
    fields = set(CatalogRecord.model_fields)
    for row in rows:
        data = {key: clean_optional_value(row.get(key)) for key in fields if key in row}
        if all(data.get(key) for key in ("video_id", "date", "source_description", "video_url")):
            records.append(CatalogRecord(**data))
    return records


def dedupe_catalog_records(records: list[CatalogRecord]) -> list[CatalogRecord]:
    output: list[CatalogRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.video_id in seen:
            continue
        output.append(record)
        seen.add(record.video_id)
    return output


def normalize_storage_video_id(video_id: str) -> str:
    normalized = re.sub(r"_mmirleb_\d+$", "", video_id)
    normalized = re.sub(r"_\d{2}$", "", normalized)
    return normalized


def synthesized_video_url(video_id: str) -> str:
    return f"{CLOUDFRONT_VIDEO_BASE_URL}/{video_id}.mp4"


def synthesized_thumbnail_url(video_id: str) -> str:
    return f"{CLOUDFRONT_THUMBNAIL_BASE_URL}/{video_id}.jpg"


def date_from_video_id(video_id: str) -> str:
    match = re.match(r"^(20\d\d-\d\d-\d\d)", video_id)
    return match.group(1) if match else "unknown"


def description_from_video_id(video_id: str) -> str:
    slug = re.sub(r"^20\d\d-\d\d-\d\d_", "", video_id)
    return slug.replace("_", " ").strip() or video_id


def append_note(existing: str, note: str) -> str:
    return f"{existing}; {note}" if existing else note


def clean_optional_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def parse_readme_catalog(readme_text: str) -> list[CatalogRecord]:
    records: list[CatalogRecord] = []
    for line in readme_text.splitlines():
        if not re.match(r"^\|\s*20\d\d-\d\d-\d\d\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        date = cells[0]
        image_cell = cells[1]
        description = cells[2]
        source_original_title = cells[3] if len(cells) >= 6 else None
        source_town = cells[4] if len(cells) >= 6 else None
        link_cell = cells[-1]
        video_url = _extract_markdown_url(link_cell)
        if not video_url:
            continue
        thumbnail_url = _extract_img_src(image_cell)
        video_id = Path(urlparse(video_url).path).stem
        warning = SOURCE_METADATA_WARNING if source_original_title or source_town else None
        records.append(
            CatalogRecord(
                video_id=video_id,
                date=date,
                source_description=description,
                video_url=video_url,
                thumbnail_url=thumbnail_url,
                source_banner_arabic_title=clean_cell(source_original_title) or None,
                source_geo_town=clean_cell(source_town) or None,
                source_metadata_warning=warning,
            )
        )
    return records


def parse_manifest_tsv(manifest_text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(manifest_text.splitlines(), delimiter="\t")
    for row in reader:
        current = clean_cell(row.get("current_stem", ""))
        target = clean_cell(row.get("target_stem", ""))
        if current:
            rows[current] = row
        if target:
            rows[target] = row
    return rows


def parse_banner_title_audit_tsv(banner_text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(banner_text.splitlines(), delimiter="\t")
    for row in reader:
        video_file = clean_cell(row.get("video_file", ""))
        video_id = Path(video_file).stem if video_file else ""
        if video_id:
            rows[video_id] = row
    return rows


def parse_geo_records_csv(geo_text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(geo_text.splitlines())
    for row in reader:
        video_url = clean_cell(row.get("video_url", ""))
        thumbnail_url = clean_cell(row.get("thumbnail_url", ""))
        source_url = video_url or thumbnail_url
        video_id = Path(urlparse(source_url).path).stem if source_url else ""
        if video_id:
            rows[video_id] = row
    return rows


def merge_source_metadata(
    record: CatalogRecord,
    manifest_rows: dict[str, dict[str, str]],
    banner_rows: dict[str, dict[str, str]],
    geo_rows: dict[str, dict[str, str]],
) -> CatalogRecord:
    update: dict[str, str | None] = {}
    manifest_row = manifest_rows.get(record.video_id)
    if manifest_row:
        update.update(
            {
                "current_stem": clean_cell(manifest_row.get("current_stem")) or None,
                "target_stem": clean_cell(manifest_row.get("target_stem")) or None,
                "manifest_confidence": clean_cell(manifest_row.get("confidence")) or None,
                "manifest_notes": clean_cell(manifest_row.get("notes")) or None,
            }
        )

    banner_row = banner_rows.get(record.video_id)
    if banner_row:
        update.update(
            {
                "source_banner_date": clean_cell(banner_row.get("banner_date")) or None,
                "source_banner_hijri_date": clean_cell(banner_row.get("hijri_date")) or None,
                "source_banner_arabic_title": clean_cell(banner_row.get("arabic_title")) or None,
                "source_banner_old_description": clean_cell(banner_row.get("old_description")) or None,
                "source_banner_new_description": clean_cell(banner_row.get("new_description")) or None,
                "source_banner_date_match": clean_cell(banner_row.get("date_match")) or None,
                "source_banner_notes": clean_cell(banner_row.get("notes")) or None,
            }
        )

    geo_row = geo_rows.get(record.video_id)
    if geo_row:
        update.update(
            {
                "source_geo_description": clean_cell(geo_row.get("description")) or None,
                "source_geo_town": clean_cell(geo_row.get("town")) or None,
                "source_geo_lat_text": clean_cell(geo_row.get("lat")) or None,
                "source_geo_lon_text": clean_cell(geo_row.get("lon")) or None,
                "source_geo_status": clean_cell(geo_row.get("status")) or None,
            }
        )

    if banner_row or geo_row:
        update["source_metadata_warning"] = SOURCE_METADATA_WARNING
    return record.model_copy(update=update) if update else record


def write_source_tables(
    output: Path,
    banner_rows: dict[str, dict[str, str]],
    geo_rows: dict[str, dict[str, str]],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if banner_rows:
        path = output / "source_banner_title_audit.parquet"
        pd.DataFrame(list(banner_rows.values())).to_parquet(path, index=False)
        paths["banner_title_audit"] = path.name
    if geo_rows:
        path = output / "source_geo_records.parquet"
        pd.DataFrame(list(geo_rows.values())).to_parquet(path, index=False)
        paths["geo_records"] = path.name
    return paths


def read_catalog(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def shortlist_catalog(catalog_path: Path, limit: int = 10) -> pd.DataFrame:
    catalog = read_catalog(catalog_path)
    columns = ["video_id", "date", "source_description", "source_banner_new_description", "video_url"]
    return catalog.loc[:, [column for column in columns if column in catalog.columns]].head(limit)


def read_text_source(source: str | Path | None) -> str:
    if source is None:
        raise ValueError("source is required")
    source_text = str(source)
    parsed = urlparse(source_text)
    if parsed.scheme in {"http", "https"}:
        request = Request(source_text, headers={"User-Agent": "fpv-vggt-visual-geometry-lab/1.0"})
        for attempt in range(3):
            try:
                with urlopen(request) as response:
                    return response.read().decode("utf-8")
            except HTTPError as exc:
                if exc.code != 429 or attempt == 2:
                    raise
                time.sleep(1.0 + attempt)
    if parsed.scheme == "file":
        with urlopen(source_text) as response:
            return response.read().decode("utf-8")
    return Path(source_text).read_text(encoding="utf-8")


def sha256_text(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def clean_cell(value: str | None) -> str:
    return (value or "").strip()


def count_readme_video_urls(readme_text: str) -> int:
    return len(re.findall(r"https?://[^\s)]+\.mp4", readme_text))


def parse_declared_video_count(readme_text: str) -> int | None:
    match = re.search(r"Video count:\s*(\d+)\s*MP4", readme_text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_markdown_url(cell: str) -> str | None:
    match = re.search(r"\[[^\]]+\]\(([^)]+)\)", cell)
    return match.group(1) if match else None


def _extract_img_src(cell: str) -> str | None:
    match = re.search(r"src=[\"']([^\"']+)[\"']", cell)
    return match.group(1) if match else None
