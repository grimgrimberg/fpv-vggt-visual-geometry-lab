from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOCAL_ONLY_ROOTS = [
    ".tmp",
    "data/catalog",
    "data/media",
    "data/frames",
    "data/features",
    "data/vggt",
    "data/geometry",
    "data/models",
    "data/annotations",
    "outputs",
]

LOCAL_ONLY_ARTIFACT_EXTENSIONS = {
    ".avi",
    ".gif",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".jsonl",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".ply",
    ".png",
    ".webm",
    ".zip",
}

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "venv",
}


def audit_local_only_artifacts(
    root: Path,
    report: Path,
    allowed_roots: list[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    allowed_roots = allowed_roots or LOCAL_ONLY_ROOTS
    allowed_parts = [_parts(entry) for entry in allowed_roots]
    flagged = []
    checked_count = 0

    for path in _iter_files(root):
        if path.suffix.lower() not in LOCAL_ONLY_ARTIFACT_EXTENSIONS:
            continue
        checked_count += 1
        relative = path.relative_to(root)
        if _under_allowed_root(relative, allowed_parts):
            continue
        flagged.append(
            {
                "relative_path": relative.as_posix(),
                "extension": path.suffix.lower(),
                "reason": "local-only artifact extension outside local-only roots",
            }
        )

    result = {
        "status": "failed_soft" if flagged else "passed",
        "root": str(root),
        "allowed_roots": allowed_roots,
        "scanned_extensions": sorted(LOCAL_ONLY_ARTIFACT_EXTENSIONS),
        "checked_count": checked_count,
        "flagged_count": len(flagged),
        "flagged": flagged,
        "warnings": [
            "filesystem placement audit only",
            "no geolocation",
            "no meters",
            "local-only media",
        ],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def _parts(path_text: str) -> tuple[str, ...]:
    return Path(path_text).parts


def _under_allowed_root(relative: Path, allowed_roots: list[tuple[str, ...]]) -> bool:
    parts = relative.parts
    return any(parts[: len(allowed)] == allowed for allowed in allowed_roots)
