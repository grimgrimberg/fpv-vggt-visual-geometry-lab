import json
from pathlib import Path

from typer.testing import CliRunner

from fpv_vggt_lab.cli import app


runner = CliRunner()


def test_review_audit_local_only_flags_media_outside_allowed_roots(tmp_path: Path):
    allowed = tmp_path / "data" / "media" / "allowed.mp4"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"allowed local media")
    suspicious = tmp_path / "leaked_frame.jpg"
    suspicious.write_bytes(b"suspicious media-derived file")
    report = tmp_path / "local_only_report.json"

    result = runner.invoke(
        app,
        [
            "review",
            "audit-local-only",
            "--root",
            str(tmp_path),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code != 0
    assert "local-only audit failed" in result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "failed_soft"
    assert data["flagged_count"] == 1
    assert data["flagged"][0]["relative_path"] == "leaked_frame.jpg"
    assert data["flagged"][0]["reason"] == "local-only artifact extension outside local-only roots"
    assert "data/media/allowed.mp4" not in json.dumps(data)


def test_review_audit_local_only_flags_manifests_and_tables_outside_allowed_roots(
    tmp_path: Path,
):
    allowed = tmp_path / "outputs" / "review" / "heatmaps.json"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("{}", encoding="utf-8")
    (tmp_path / "leaked_heatmaps.json").write_text("{}", encoding="utf-8")
    (tmp_path / "leaked_catalog.parquet").write_bytes(b"not real parquet")
    report = tmp_path / "local_only_report.json"

    result = runner.invoke(
        app,
        [
            "review",
            "audit-local-only",
            "--root",
            str(tmp_path),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code != 0
    data = json.loads(report.read_text(encoding="utf-8"))
    flagged = {row["relative_path"] for row in data["flagged"]}
    assert flagged == {"leaked_heatmaps.json", "leaked_catalog.parquet"}


def test_review_audit_local_only_passes_when_artifacts_are_under_local_roots(tmp_path: Path):
    allowed = tmp_path / "outputs" / "review" / "local_review.html"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("<html></html>", encoding="utf-8")
    report = tmp_path / "local_only_report.json"

    result = runner.invoke(
        app,
        [
            "review",
            "audit-local-only",
            "--root",
            str(tmp_path),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["flagged_count"] == 0
