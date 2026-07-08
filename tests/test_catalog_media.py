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


def test_catalog_sync_preserves_latest_source_metadata_as_isolated_fields(tmp_path: Path):
    readme, manifest, video_id = write_fixture_catalog(tmp_path)
    banner = tmp_path / "banner.tsv"
    banner.write_text(
        "\n".join(
            [
                "video_file\treadme_date\tbanner_date\thijri_date\tarabic_title\told_description\tnew_description\tdate_match\tnotes",
                (
                    f"{video_id}.mp4\t2026-06-22\t2026-06-22\t01 Muharram 1448\t"
                    "fixture arabic title\told fixture\tNew fixture source description\tyes\tkept as source metadata"
                ),
            ]
        ),
        encoding="utf-8",
    )
    geo = tmp_path / "geo.csv"
    geo.write_text(
        "\n".join(
            [
                "row,date,description,town,lat,lon,status,thumbnail_url,video_url",
                (
                    "1,2026-06-22,Fixture geo description,Fixture Town,33.0,35.0,mapped,"
                    f"https://example.test/{video_id}.jpg,{(tmp_path / f'{video_id}.mp4').as_uri()}"
                ),
            ]
        ),
        encoding="utf-8",
    )
    website = tmp_path / "site.html"
    website.write_text("<title>FPV Drone-Strike Dataset Viewer</title>", encoding="utf-8")
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
            "--banner-audit",
            str(banner),
            "--geo-records",
            str(geo),
            "--website",
            str(website),
            "--output",
            str(catalog_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = pd.read_parquet(catalog_dir / "catalog.parquet")
    row = catalog.iloc[0]
    assert row["source_banner_new_description"] == "New fixture source description"
    assert row["source_geo_town"] == "Fixture Town"
    assert row["source_geo_lat_text"] == "33.0"
    assert "source metadata only" in row["source_metadata_warning"]
    assert (catalog_dir / "source_banner_title_audit.parquet").exists()
    assert (catalog_dir / "source_geo_records.parquet").exists()

    snapshot = json.loads((catalog_dir / "catalog_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["banner_audit_sha256"]
    assert snapshot["geo_records_sha256"]
    assert snapshot["website_sha256"]
    assert snapshot["source_table_paths"] == {
        "banner_title_audit": "source_banner_title_audit.parquet",
        "geo_records": "source_geo_records.parquet",
    }
    assert "geolocation" in snapshot["source_metadata_warnings"][0]


def test_catalog_sync_reconciles_geo_manifest_and_fallback_to_declared_count(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("Storage Video count: 3 MP4 files\n", encoding="utf-8")

    geo_id = "2026-07-01_current_high_quality_mmirleb_100"
    geo = tmp_path / "geo.csv"
    geo.write_text(
        "\n".join(
            [
                "row,date,description,town,lat,lon,status,thumbnail_url,video_url",
                (
                    "1,2026-07-01,Current high quality clip,Source Town,33.0,35.0,mapped,"
                    f"https://d2fioemadmrru3.cloudfront.net/thumbnails/{geo_id}.jpg,"
                    f"https://d2fioemadmrru3.cloudfront.net/videos/{geo_id}.mp4"
                ),
            ]
        ),
        encoding="utf-8",
    )

    manifest_only_id = "2026-07-02_manifest_only_clip"
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join(
            [
                "current_stem\ttarget_stem\tdate\tslug\tconfidence\tnotes",
                f"{manifest_only_id}\t{manifest_only_id}\t2026-07-02\tmanifest_only_clip\thigh\tmanifest-only storage row",
            ]
        ),
        encoding="utf-8",
    )

    fallback_path = tmp_path / "fallback.parquet"
    pd.DataFrame(
        [
            {
                "video_id": "2026-07-01_current_high_quality_mmirleb_102",
                "date": "2026-07-01",
                "source_description": "Old replacement duplicate that should be skipped",
                "video_url": "https://d2fioemadmrru3.cloudfront.net/videos/2026-07-01_current_high_quality_mmirleb_102.mp4",
                "thumbnail_url": "https://d2fioemadmrru3.cloudfront.net/thumbnails/2026-07-01_current_high_quality_mmirleb_102.jpg",
                "manifest_confidence": "old-high",
                "manifest_notes": "old duplicate",
            },
            {
                "video_id": "2026-07-03_missing_storage_clip",
                "date": "2026-07-03",
                "source_description": "Missing storage clip",
                "video_url": "https://d2fioemadmrru3.cloudfront.net/videos/2026-07-03_missing_storage_clip.mp4",
                "thumbnail_url": "https://d2fioemadmrru3.cloudfront.net/thumbnails/2026-07-03_missing_storage_clip.jpg",
                "manifest_confidence": "old-high",
                "manifest_notes": "previous snapshot row",
            },
        ]
    ).to_parquet(fallback_path, index=False)

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
            "--geo-records",
            str(geo),
            "--fallback-catalog",
            str(fallback_path),
            "--output",
            str(catalog_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = pd.read_parquet(catalog_dir / "catalog.parquet")
    assert len(catalog) == 3
    assert set(catalog["video_id"]) == {
        geo_id,
        manifest_only_id,
        "2026-07-03_missing_storage_clip",
    }
    assert "2026-07-01_current_high_quality_mmirleb_102" not in set(catalog["video_id"])
    manifest_only = catalog[catalog["video_id"] == manifest_only_id].iloc[0]
    assert manifest_only["video_url"].endswith(f"/{manifest_only_id}.mp4")
    fallback = catalog[catalog["video_id"] == "2026-07-03_missing_storage_clip"].iloc[0]
    assert fallback["manifest_confidence"] == "fallback_previous_catalog"
    assert "source metadata only" in fallback["source_metadata_warning"]

    snapshot = json.loads((catalog_dir / "catalog_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["row_count"] == 3
    assert snapshot["readme_table_row_count"] == 0
    assert snapshot["upstream_declared_video_count"] == 3
    assert any("previous local catalog" in warning for warning in snapshot["source_metadata_warnings"])
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
