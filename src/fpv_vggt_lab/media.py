from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname, urlopen

import cv2
import pandas as pd

from .schemas import MediaInventoryRecord, model_to_dict


CheckStatus = Literal["pass", "warn", "fail"]
MEDIA_AUDIT_WARNINGS = [
    "local-only media",
    "no geolocation",
    "no meters",
    "workflow readiness only",
]


def fetch_media(
    catalog_path: Path,
    media_dir: Path,
    inventory_path: Path,
    video_id: str,
) -> MediaInventoryRecord:
    catalog = pd.read_parquet(catalog_path)
    matches = catalog[catalog["video_id"] == video_id]
    if matches.empty:
        raise ValueError(f"video_id not found in catalog: {video_id}")
    row = matches.iloc[0]
    source_url = str(row["video_url"])

    media_dir = media_dir.resolve()
    media_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(source_url).path).suffix or ".mp4"
    local_path = media_dir / f"{video_id}{suffix}"
    copy_source_to_path(source_url, local_path)

    probe = probe_video(local_path)
    record = MediaInventoryRecord(
        video_id=video_id,
        source_url=source_url,
        local_path=local_path.resolve(),
        sha256=sha256_file(local_path),
        bytes=local_path.stat().st_size,
        **probe,
    )
    upsert_inventory(inventory_path, record)
    return record


def read_media_inventory(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def find_media_record(inventory_path: Path, video_id: str) -> MediaInventoryRecord:
    inventory = read_media_inventory(inventory_path)
    matches = inventory[inventory["video_id"] == video_id]
    if matches.empty:
        raise ValueError(f"video_id not found in media inventory: {video_id}")
    return MediaInventoryRecord.model_validate(matches.iloc[0].to_dict())


def audit_media_inventory(
    inventory_path: Path,
    video_ids: list[str] | None = None,
) -> dict[str, Any]:
    inventory = read_media_inventory(inventory_path)
    if video_ids:
        selected = inventory[inventory["video_id"].isin(video_ids)]
        missing = sorted(set(video_ids) - set(selected["video_id"].astype(str)))
    else:
        selected = inventory
        missing = []

    clips = [_audit_media_row(row.to_dict()) for _index, row in selected.iterrows()]
    for video_id in missing:
        clips.append(
            {
                "video_id": video_id,
                "status": "needs_review",
                "ready": False,
                "media_path": None,
                "checks": {
                    "inventory_row": _check("fail", ["video_id not found in media inventory"])
                },
            }
        )
    ready_count = sum(1 for clip in clips if clip["ready"])
    status = "ready" if clips and ready_count == len(clips) else "needs_review"
    return {
        "status": status,
        "checked_count": len(clips),
        "ready_count": ready_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": MEDIA_AUDIT_WARNINGS,
        "interpretation": (
            "Local media workflow readiness only. This is not a claim about location, "
            "scale, target identity, authenticity, or operational usefulness."
        ),
        "clips": clips,
    }


def write_media_audit_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def copy_source_to_path(source_url: str, local_path: Path) -> None:
    parsed = urlparse(source_url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if parsed.scheme == "file":
        shutil.copyfile(Path(url2pathname(parsed.path)), local_path)
    elif parsed.scheme in {"http", "https"}:
        with urlopen(source_url) as response, local_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    elif parsed.scheme == "":
        shutil.copyfile(Path(source_url), local_path)
    else:
        raise ValueError(f"unsupported media URL scheme: {parsed.scheme}")


def _audit_media_row(row: dict[str, Any]) -> dict[str, Any]:
    video_id = str(row.get("video_id", ""))
    local_path = Path(str(row.get("local_path", "")))
    checks: dict[str, dict[str, Any]] = {}

    exists = local_path.exists()
    checks["file_present"] = _check(
        "pass" if exists else "fail",
        [] if exists else [f"local media path missing: {local_path}"],
        {"path": str(local_path)},
    )

    if exists:
        actual_sha256 = sha256_file(local_path)
        expected_sha256 = str(row.get("sha256", ""))
        checksum_messages = [] if actual_sha256 == expected_sha256 else ["sha256 mismatch"]
        checks["checksum"] = _check(
            "fail" if checksum_messages else "pass",
            checksum_messages,
            {"expected_sha256": expected_sha256, "actual_sha256": actual_sha256},
        )
        actual_bytes = local_path.stat().st_size
        expected_bytes = int(row.get("bytes", -1))
        byte_messages = [] if actual_bytes == expected_bytes else ["byte count mismatch"]
        checks["file_size"] = _check(
            "fail" if byte_messages else "pass",
            byte_messages,
            {"expected_bytes": expected_bytes, "actual_bytes": actual_bytes},
        )
        probe = _decode_probe(
            local_path,
            frame_count=int(row.get("frame_count", 0)),
            fps=float(row.get("fps", 0) or 0),
        )
        if not probe["opened"]:
            probe_status: CheckStatus = "fail"
            probe_messages = ["could not decode video"]
        elif probe["failed_reads"]:
            probe_status = "warn"
            probe_messages = [f"decode probe failures: {probe['failed_reads']}/3"]
        else:
            probe_status = "pass"
            probe_messages = []
        checks["decode_probe"] = _check(probe_status, probe_messages, probe)
    else:
        checks["checksum"] = _check("fail", ["media file unavailable for checksum"])
        checks["file_size"] = _check("fail", ["media file unavailable for size check"])
        checks["decode_probe"] = _check("fail", ["media file unavailable for decode probe"])

    metadata_messages = []
    if float(row.get("fps", 0) or 0) <= 0:
        metadata_messages.append("inventory fps is not positive")
    if int(row.get("frame_count", 0)) <= 0:
        metadata_messages.append("inventory frame_count is zero")
    if int(row.get("width", 0)) <= 0 or int(row.get("height", 0)) <= 0:
        metadata_messages.append("inventory dimensions are zero")
    checks["inventory_metadata"] = _check(
        "fail" if metadata_messages else "pass",
        metadata_messages,
        {
            "duration_sec": float(row.get("duration_sec", 0) or 0),
            "fps": float(row.get("fps", 0) or 0),
            "frame_count": int(row.get("frame_count", 0)),
            "width": int(row.get("width", 0)),
            "height": int(row.get("height", 0)),
        },
    )

    ready = all(check["status"] == "pass" for check in checks.values())
    return {
        "video_id": video_id,
        "status": "ready" if ready else "needs_review",
        "ready": ready,
        "media_path": str(local_path),
        "checks": checks,
    }


def _decode_probe(video_path: Path, frame_count: int, fps: float) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    reads = []
    try:
        opened = capture.isOpened()
        probed_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
        effective_frame_count = probed_frame_count if probed_frame_count > 0 else frame_count
        last_index = max(effective_frame_count - 1, 0)
        positions = [
            0,
            max(int(round(last_index * 0.5)), 0),
            max(int(round(last_index * 0.9)), 0),
        ]
        for position in positions:
            ok = False
            if opened and effective_frame_count > 0 and fps > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, _frame = capture.read()
            reads.append({"frame_index": int(position), "ok": bool(ok)})
    finally:
        capture.release()
    return {
        "path": str(video_path),
        "opened": bool(reads and any(read["ok"] for read in reads)),
        "inventory_frame_count": int(frame_count),
        "probe_frame_count": int(probed_frame_count) if "probed_frame_count" in locals() else 0,
        "reads": reads,
        "failed_reads": sum(1 for read in reads if not read["ok"]),
    }


def _check(status: CheckStatus, messages: list[str], details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": status, "messages": messages, "details": details or {}}


def probe_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0:
        fps = 1.0
    duration = frame_count / fps if fps else 0.0
    return {
        "duration_sec": float(duration),
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upsert_inventory(path: Path, record: MediaInventoryRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([model_to_dict(record)])
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[existing["video_id"] != record.video_id]
        data = pd.concat([existing, new_row], ignore_index=True)
    else:
        data = new_row
    data.to_parquet(path, index=False)
