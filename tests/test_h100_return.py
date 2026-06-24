import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from tests.test_viz_compare_smoothing import create_mocked_summary


runner = CliRunner()


def make_fake_h100_return(
    tmp_path: Path,
    *,
    video_id: str = "h100-return-video",
    forbidden_feature: bool = False,
) -> Path:
    _frame_manifest, bundle, _summary, _video = create_mocked_summary(tmp_path, video_id)
    root = tmp_path / f"{video_id}_return"
    selected = root / "selected" / video_id / "segment-001"
    shutil.copytree(bundle, selected)

    features = root / "features"
    features.mkdir(parents=True)
    feature_row = {
        "video_id": video_id,
        "segment_id": "segment-001",
        "reconstruction_reliability_score": 0.91,
        "point_count": 30,
    }
    if forbidden_feature:
        feature_row["target_class"] = "vehicle"
    pd.DataFrame([feature_row]).to_parquet(features / "segment_features.parquet", index=False)

    selected_report = pd.DataFrame(
        [
            {
                "video_id": video_id,
                "segment_id": "segment-001",
                "selected_tier": "main",
                "status": "selected",
                "reason": "fake tested return",
                "bundle_path": f"selected/{video_id}/segment-001",
            }
        ]
    )
    selected_report.to_parquet(root / "selected_bundle_report.parquet", index=False)
    (root / "cloud_summary.json").write_text(
        json.dumps(
            {
                "status": "done_partial",
                "clips": [
                    {
                        "video_id": video_id,
                        "segment_id": "segment-001",
                        "tier": "main",
                        "status": "done",
                        "bundle_valid": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "run.log").write_text("fake h100 run\n", encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps({"python": "test", "gpu_name": "NVIDIA H100 test double"}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "h100-return-v1",
                "run_id": "fake-h100-run",
                "status": "done_partial",
                "warnings": [
                    "no geolocation",
                    "no meters",
                    "relative VGGT frame",
                    "local-only media",
                ],
                "selected_bundle_count": 1,
                "feature_files": ["features/segment_features.parquet"],
                "selected_bundle_report": "selected_bundle_report.parquet",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    zip_path = tmp_path / f"{video_id}_h100_return.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root))
    return zip_path


def test_h100_inspect_and_import_return(tmp_path: Path):
    source = make_fake_h100_return(tmp_path)
    workdir = tmp_path / "h100-workdir"
    vggt_root = tmp_path / "data" / "vggt"
    review_output = tmp_path / "reviews" / "h100"

    inspected = runner.invoke(app, ["h100", "inspect-return", "--source", str(source)])
    assert inspected.exit_code == 0, inspected.output
    assert "H100 return inspection" in inspected.output
    assert "selected bundles: 1" in inspected.output
    assert "status: done_partial" in inspected.output

    dry_run = runner.invoke(
        app,
        [
            "h100",
            "import-return",
            "--source",
            str(source),
            "--workdir",
            str(workdir),
            "--vggt-root",
            str(vggt_root),
            "--review-output",
            str(review_output),
            "--dry-run",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    dry_report = json.loads((review_output / "import_report.json").read_text(encoding="utf-8"))
    assert dry_report["status"] == "done"
    assert dry_report["dry_run"] is True
    assert dry_report["would_import_count"] == 1
    assert not vggt_root.exists()

    imported = runner.invoke(
        app,
        [
            "h100",
            "import-return",
            "--source",
            str(source),
            "--workdir",
            str(workdir),
            "--vggt-root",
            str(vggt_root),
            "--review-output",
            str(review_output),
        ],
    )
    assert imported.exit_code == 0, imported.output
    report = json.loads((review_output / "import_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert report["imported_count"] == 1
    assert (vggt_root / "h100-return-video" / "segment-001" / "metadata.json").exists()
    assert (review_output / "index.html").exists()
    assert (review_output / "summary.json").exists()
    assert (review_output / "NEXT_STEPS.md").exists()
    html = (review_output / "index.html").read_text(encoding="utf-8")
    assert "H100 Import Review" in html
    assert "No geolocation" in html


def test_h100_import_return_rejects_forbidden_feature_columns(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="h100-forbidden-feature-video",
        forbidden_feature=True,
    )
    review_output = tmp_path / "reviews" / "h100"
    result = runner.invoke(
        app,
        [
            "h100",
            "import-return",
            "--source",
            str(source),
            "--workdir",
            str(tmp_path / "workdir"),
            "--vggt-root",
            str(tmp_path / "data" / "vggt"),
            "--review-output",
            str(review_output),
        ],
    )

    assert result.exit_code != 0
    report = json.loads((review_output / "import_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed_soft"
    assert any("target_class" in issue for issue in report["issues"])
    assert not (tmp_path / "data" / "vggt").exists()
