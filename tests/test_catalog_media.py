import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.synthetic import create_synthetic_video


runner = CliRunner()


def write_fixture_catalog(tmp_path: Path) -> tuple[Path, Path, str]:
    video_id = "2026-06-22_fixture_reconstruction_clip"
    video_path = tmp_path / f"{video_id}.mp4"
    create_synthetic_video(video_path, frames=24, width=160, height=120, fps=12)

    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Fixture Dataset",
                "",
                "| Date | Image | Description | Link |",
                "|---|---|---|---|",
                (
                    f"| 2026-06-22 | <img src=\"https://example.test/thumb.jpg\" "
                    f"alt=\"Fixture\" width=\"180\"> | Fixture reconstruction clip | "
                    f"[Download]({video_path.as_uri()}) |"
                ),
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join(
            [
                "current_stem\ttarget_stem\tdate\tslug\tconfidence\tnotes",
                (
                    f"{video_id}\t{video_id}\t2026-06-22\t"
                    "fixture_reconstruction_clip\thigh\tFixture row."
                ),
            ]
        ),
        encoding="utf-8",
    )
    return readme, manifest, video_id


def test_catalog_sync_pins_readme_and_manifest_snapshot(tmp_path: Path):
    readme, manifest, video_id = write_fixture_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"

    result = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    catalog_path = catalog_dir / "catalog.parquet"
    snapshot_path = catalog_dir / "catalog_snapshot.json"
    assert catalog_path.exists()
    assert snapshot_path.exists()

    catalog = pd.read_parquet(catalog_path)
    assert list(catalog["video_id"]) == [video_id]
    assert list(catalog["source_description"]) == ["Fixture reconstruction clip"]
    assert list(catalog["manifest_confidence"]) == ["high"]
    assert str(catalog.loc[0, "video_url"]).startswith("file:///")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["row_count"] == 1
    assert snapshot["readme_sha256"]
    assert snapshot["manifest_sha256"]


def test_media_fetch_is_explicit_and_records_checksum_inventory(tmp_path: Path):
    readme, manifest, video_id = write_fixture_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    media_dir = tmp_path / "media"
    inventory_path = tmp_path / "media_inventory.parquet"

    sync = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )
    assert sync.exit_code == 0, sync.output

    missing_id = runner.invoke(
        app,
        [
            "media",
            "fetch",
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(media_dir),
            "--inventory",
            str(inventory_path),
        ],
    )
    assert missing_id.exit_code != 0
    assert "--video-id" in missing_id.output

    fetched = runner.invoke(
        app,
        [
            "media",
            "fetch",
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(media_dir),
            "--inventory",
            str(inventory_path),
            "--video-id",
            video_id,
        ],
    )
    assert fetched.exit_code == 0, fetched.output

    inventory = pd.read_parquet(inventory_path)
    assert list(inventory["video_id"]) == [video_id]
    local_path = Path(inventory.loc[0, "local_path"])
    assert local_path.exists()
    assert inventory.loc[0, "sha256"]
    assert inventory.loc[0, "bytes"] > 0
    assert inventory.loc[0, "frame_count"] >= 24
    assert inventory.loc[0, "width"] == 160
    assert inventory.loc[0, "height"] == 120


def test_media_audit_reports_clean_cached_video(tmp_path: Path):
    readme, manifest, video_id = write_fixture_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    media_dir = tmp_path / "media"
    inventory_path = tmp_path / "media_inventory.parquet"
    report_path = tmp_path / "media_audit.json"

    sync = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )
    assert sync.exit_code == 0, sync.output
    fetched = runner.invoke(
        app,
        [
            "media",
            "fetch",
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(media_dir),
            "--inventory",
            str(inventory_path),
            "--video-id",
            video_id,
        ],
    )
    assert fetched.exit_code == 0, fetched.output

    audit = runner.invoke(
        app,
        [
            "media",
            "audit",
            "--inventory",
            str(inventory_path),
            "--video-id",
            video_id,
            "--report",
            str(report_path),
        ],
    )

    assert audit.exit_code == 0, audit.output
    assert "media audit ready" in audit.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["ready_count"] == 1
    clip = report["clips"][0]
    assert clip["video_id"] == video_id
    assert clip["status"] == "ready"
    assert clip["checks"]["file_present"]["status"] == "pass"
    assert clip["checks"]["checksum"]["status"] == "pass"
    assert clip["checks"]["decode_probe"]["status"] == "pass"
    assert "not a claim about location" in report["interpretation"]


def test_media_audit_flags_checksum_mismatch(tmp_path: Path):
    readme, manifest, video_id = write_fixture_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    media_dir = tmp_path / "media"
    inventory_path = tmp_path / "media_inventory.parquet"
    report_path = tmp_path / "media_audit.json"

    sync = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )
    assert sync.exit_code == 0, sync.output
    fetched = runner.invoke(
        app,
        [
            "media",
            "fetch",
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(media_dir),
            "--inventory",
            str(inventory_path),
            "--video-id",
            video_id,
        ],
    )
    assert fetched.exit_code == 0, fetched.output
    inventory = pd.read_parquet(inventory_path)
    create_synthetic_video(
        Path(inventory.loc[0, "local_path"]),
        frames=12,
        width=160,
        height=120,
        fps=12,
    )

    audit = runner.invoke(
        app,
        [
            "media",
            "audit",
            "--inventory",
            str(inventory_path),
            "--video-id",
            video_id,
            "--report",
            str(report_path),
        ],
    )

    assert audit.exit_code != 0
    assert "needs_review" in audit.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "needs_review"
    clip = report["clips"][0]
    assert clip["checks"]["checksum"]["status"] == "fail"
    assert "sha256 mismatch" in clip["checks"]["checksum"]["messages"]
    assert clip["checks"]["decode_probe"]["status"] in {"pass", "warn"}


def test_media_audit_tolerates_container_tail_frame_imprecision(tmp_path: Path):
    readme, manifest, video_id = write_fixture_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    media_dir = tmp_path / "media"
    inventory_path = tmp_path / "media_inventory.parquet"
    report_path = tmp_path / "media_audit.json"

    sync = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )
    assert sync.exit_code == 0, sync.output
    fetched = runner.invoke(
        app,
        [
            "media",
            "fetch",
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(media_dir),
            "--inventory",
            str(inventory_path),
            "--video-id",
            video_id,
        ],
    )
    assert fetched.exit_code == 0, fetched.output
    inventory = pd.read_parquet(inventory_path)
    inventory.loc[0, "frame_count"] = int(inventory.loc[0, "frame_count"]) + 1
    inventory.to_parquet(inventory_path, index=False)

    audit = runner.invoke(
        app,
        [
            "media",
            "audit",
            "--inventory",
            str(inventory_path),
            "--video-id",
            video_id,
            "--report",
            str(report_path),
        ],
    )

    assert audit.exit_code == 0, audit.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    clip = report["clips"][0]
    assert clip["checks"]["decode_probe"]["status"] == "pass"
    probed_indices = [
        row["frame_index"] for row in clip["checks"]["decode_probe"]["details"]["reads"]
    ]
    assert max(probed_indices) < int(inventory.loc[0, "frame_count"]) - 1
