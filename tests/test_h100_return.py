import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.h100_return import validate_h100_return, verify_h100_return_against_launch
from fpv_vggt_lab.method_contract import expanded_method_matrix, write_method_contract_files
from fpv_vggt_lab.review import run_three_clip_review
from tests.test_viz_compare_smoothing import create_mocked_summary


runner = CliRunner()


def write_inner_zip(path: Path, name: str, content: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)


def make_fake_launch_manifest(tmp_path: Path, *, required_files: list[str] | None = None) -> Path:
    required = required_files or [
        "manifest.json",
        "cloud_run.log",
        "environment.json",
        "expanded_runtime_check.json",
        "method_matrix.json",
        "return_contract.json",
        "method_stage_plan.json",
        "method_stage_report.json",
    ]
    path = tmp_path / "runpod_launch_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "runpod-launch-manifest-v1",
                "status": "ready_to_upload",
                "package": {"sha256": "fake-package-sha", "clip_count": 1},
                "methods": [method["method_id"] for method in expanded_method_matrix()],
                "expected_return": {
                    "primary_download": "h100_return.zip",
                    "required_files_inside_return": required,
                    "expanded_artifacts_when_available": [
                        "colmap_sparse.zip",
                        "trajectories.zip",
                        "evo_reports.zip",
                    ],
                },
                "safety_warnings": [
                    "no geolocation",
                    "no meters",
                    "relative VGGT frame",
                    "local-only media",
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def make_emergency_h100_return(tmp_path: Path) -> Path:
    root = tmp_path / "emergency_return"
    root.mkdir()
    (root / "emergency_return_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "h100-emergency-return-v1",
                "status": "failed_soft",
                "reason": "70_package_return.py failed or did not create h100_return.zip",
                "warnings": [
                    "no geolocation",
                    "no meters",
                    "relative VGGT frame",
                    "local-only media",
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "cloud_run.log").write_text("cloud log before packager failure\n", encoding="utf-8")
    (root / "environment.json").write_text(json.dumps({"python": "test"}), encoding="utf-8")
    (root / "expanded_runtime_check.json").write_text(
        json.dumps(
            {
                "status": "done",
                "readiness_status": "blocked_primary_runtime",
                "checks": {"vggt_feedforward_ready": False},
                "missing_primary_checks": ["vggt_feedforward_ready"],
                "next_actions": ["Use the pinned VGGT image."],
            }
        ),
        encoding="utf-8",
    )
    failure_dir = root / "failure_packages"
    failure_dir.mkdir()
    (failure_dir / "expanded_method_stage_error.json").write_text(
        json.dumps({"status": "failed_soft", "error": "test failure"}),
        encoding="utf-8",
    )
    zip_path = tmp_path / "emergency_h100_return.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root).as_posix())
    return zip_path


def make_fake_h100_return(
    tmp_path: Path,
    *,
    video_id: str = "h100-return-video",
    forbidden_feature: bool = False,
    expanded_artifacts: bool = False,
    corrupt_artifact: bool = False,
    optional_manifests: bool = False,
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
    (root / "cloud_run.log").write_text("fake cloud log\n", encoding="utf-8")
    write_method_contract_files(root)
    (root / "environment.json").write_text(
        json.dumps({"python": "test", "gpu_name": "NVIDIA H100 test double"}),
        encoding="utf-8",
    )
    (root / "expanded_runtime_check.json").write_text(
        json.dumps({"status": "done", "checks": {"vggt_feedforward_ready": True, "vggt_colmap_ba_ready": False}}),
        encoding="utf-8",
    )
    (root / "method_stage_report.json").write_text(
        json.dumps(
            {
                "status": "done_partial",
                "stages": [
                    {"method_id": "vggt_colmap_ba_windowed", "status": "skipped_missing_dependency", "reason": "test fixture"},
                    {"method_id": "trajectory_preprocess_evo_eval", "status": "done"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "quality_report.json").write_text(json.dumps({"status": "done_partial"}), encoding="utf-8")
    if expanded_artifacts:
        if corrupt_artifact:
            (root / "colmap_sparse.zip").write_bytes(b"not a zip")
        else:
            write_inner_zip(root / "colmap_sparse.zip", "scene/sparse/README.txt", "fake sparse")
        write_inner_zip(root / "trajectories.zip", "trajectory.tum", "0 0 0 0 0 0 0 1\n")
        write_inner_zip(root / "evo_reports.zip", "README.md", "relative consistency only\n")
    if optional_manifests:
        optional_payloads = {
            "odm_relative_artifacts_manifest.json": {
                "method_id": "odm_offline_photogrammetry_baseline",
                "status": "skipped_missing_dependency",
                "selected_scene_count": 0,
                "dependency_probe": {"odm": False, "odm_run": False},
                "execution_policy": "relative only",
            },
            "showcase_manifest.json": {
                "method_id": "gsplat_nerfstudio_showcase",
                "status": "ready_pending_execution",
                "selected_scene_count": 2,
                "dependency_probe": {"nerfstudio_module": True, "gsplat_module": False},
                "execution_policy": "top-ranked windows only",
            },
            "depth_overlay_manifest.json": {
                "method_id": "relative_depth_overlay_optional",
                "status": "skipped_missing_dependency",
                "selected_scene_count": 0,
                "dependency_probe": {"transformers_module": False},
            },
            "research_methods_manifest.json": {
                "method_id": "mast3r_dust3r_vggetr_single_run",
                "status": "ready_pending_method_command",
                "selected_scene_count": 1,
                "dependency_probe": {"mast3r_module": True, "dust3r_module": False, "vggetr_module": False},
                "method_output_contract": "convert into compact bundle before review",
            },
            "vggetr_feasibility_report.json": {
                "method_id": "vggetr_candidate_unresolved",
                "status": "needs_identifier_confirmation",
                "dependency_probe": {"vggetr_module": False},
            },
        }
        for filename, payload in optional_payloads.items():
            (root / filename).write_text(json.dumps(payload), encoding="utf-8")
        (root / "candidate_identity.md").write_text(
            "# VGGeTR / VG2GT Candidate Identity\n\nStatus: configured slot. Confirm repository/checkpoint/license and command template first.\n",
            encoding="utf-8",
        )
        for name in [
            "odm_artifacts.zip",
            "odm_project_reports.zip",
            "odm_logs.zip",
            "nerfstudio_gsplat_showcase.zip",
            "nerfstudio_projects.zip",
            "gsplat_exports.zip",
            "showcase_renders.zip",
            "relative_depth_overlays.zip",
            "research_methods.zip",
            "research_methods_outputs.zip",
        ]:
            write_inner_zip(root / name, "README.md", f"{name} fixture\n")
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


def test_h100_emergency_return_is_explicit_failed_soft(tmp_path: Path):
    source = make_emergency_h100_return(tmp_path)

    report = validate_h100_return(source, extract_root=tmp_path / "emergency_extract")

    assert report["is_emergency_return"] is True
    assert report["manifest"]["schema_version"] == "h100-emergency-return-v1"
    assert report["selected_bundle_count"] == 0
    assert any("emergency return package" in issue for issue in report["issues"])
    assert "missing manifest.json" not in report["issues"]
    assert "no selected VGGT bundles found" not in report["issues"]
    assert any("No selected VGGT bundles" in action for action in report["next_actions"])

    inspected = runner.invoke(app, ["h100", "inspect-return", "--source", str(source)])
    assert inspected.exit_code != 0
    assert "schema: h100-emergency-return-v1" in inspected.output
    assert "emergency return: yes" in inspected.output
    assert "70_package_return.py failed" in inspected.output


def test_h100_inspect_and_import_return(tmp_path: Path):
    source = make_fake_h100_return(tmp_path, expanded_artifacts=True)
    workdir = tmp_path / "h100-workdir"
    vggt_root = tmp_path / "data" / "vggt"
    review_output = tmp_path / "reviews" / "h100"

    inspected = runner.invoke(app, ["h100", "inspect-return", "--source", str(source)])
    assert inspected.exit_code == 0, inspected.output
    assert "H100 return inspection" in inspected.output
    assert "selected bundles: 1" in inspected.output
    assert "expanded artifacts: 3" in inspected.output
    assert "artifact colmap_sparse.zip:" in inspected.output
    assert "status: done_partial" in inspected.output
    assert "runtime check: done" in inspected.output
    assert "runtime vggt_colmap_ba_ready: False" in inspected.output
    assert "method stage: done_partial" in inspected.output
    assert "method vggt_colmap_ba_windowed: skipped_missing_dependency" in inspected.output
    assert "method artifacts vggt_feedforward_full: usable_compact_partial" in inspected.output
    assert "method artifacts vggt_colmap_ba_windowed: artifact_partial" in inspected.output
    assert "next actions:" in inspected.output
    assert "VGGT COLMAP BA was not ready" in inspected.output

    inspected_json = runner.invoke(app, ["h100", "inspect-return", "--source", str(source), "--json"])
    assert inspected_json.exit_code == 0, inspected_json.output
    inspected_payload = json.loads(inspected_json.output)
    assert inspected_payload["expanded_runtime_check"]["checks"]["vggt_colmap_ba_ready"] is False
    assert [artifact["path"] for artifact in inspected_payload["expanded_artifacts"]] == [
        "colmap_sparse.zip",
        "evo_reports.zip",
        "trajectories.zip",
    ]
    assert inspected_payload["method_stage_report"]["stages"][0]["method_id"] == "vggt_colmap_ba_windowed"
    audit = {row["method_id"]: row for row in inspected_payload["method_artifact_audit"]}
    assert audit["vggt_feedforward_full"]["status"] == "usable_compact_partial"
    assert audit["vggt_feedforward_full"]["substitutions"] == [
        {"expected": "compact_bundles.zip", "satisfied_by": "selected/"},
        {"expected": "quality_report.parquet", "satisfied_by": "quality_report.json"},
    ]
    assert audit["vggt_colmap_ba_windowed"]["status"] == "artifact_partial"
    assert "converted_colmap_bundles.zip" in audit["vggt_colmap_ba_windowed"]["missing_artifacts"]
    assert audit["trajectory_preprocess_evo_eval"]["status"] == "artifact_partial"
    assert any("VGGT COLMAP BA was not ready" in action for action in inspected_payload["next_actions"])

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
    assert dry_report["expanded_runtime_check"]["checks"]["vggt_feedforward_ready"] is True
    assert dry_report["method_stage_report"]["status"] == "done_partial"
    assert dry_report["expanded_artifacts"][0]["path"] == "colmap_sparse.zip"
    assert dry_report["method_artifact_audit"][0]["method_id"] == "vggt_feedforward_full"
    assert any("VGGT COLMAP BA was not ready" in action for action in dry_report["next_actions"])
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
    assert (review_output / "expanded_runtime_check.json").exists()
    assert (review_output / "method_stage_report.json").exists()
    assert (review_output / "quality_report.json").exists()
    assert (review_output / "expanded_artifacts" / "colmap_sparse.zip").exists()
    assert (review_output / "expanded_artifacts" / "trajectories.zip").exists()
    assert (review_output / "expanded_artifacts" / "evo_reports.zip").exists()
    summary = json.loads((review_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["expanded_artifact_count"] == 3
    assert summary["expanded_artifact_dir"] == "expanded_artifacts"
    assert summary["method_artifact_status_counts"]["artifact_partial"] == 2
    assert summary["method_artifact_status_counts"]["usable_compact_partial"] == 1
    html = (review_output / "index.html").read_text(encoding="utf-8")
    assert "H100 Import Review" in html
    assert "No geolocation" in html
    assert "Expanded Runtime" in html
    assert "Expanded Method Stage" in html
    assert "Expanded Artifacts" in html
    assert "Method Artifact Audit" in html
    assert "colmap_sparse.zip" in html
    assert "vggt_colmap_ba_windowed" in html



def test_h100_optional_method_reports_surface_in_inspect_import_and_postflight(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="h100-optional-method-video",
        expanded_artifacts=True,
        optional_manifests=True,
    )

    inspected_json = runner.invoke(app, ["h100", "inspect-return", "--source", str(source), "--json"])
    assert inspected_json.exit_code == 0, inspected_json.output
    inspected_payload = json.loads(inspected_json.output)
    reports = {item["method_id"]: item for item in inspected_payload["optional_method_reports"]}
    assert reports["odm_offline_photogrammetry_baseline"]["status"] == "skipped_missing_dependency"
    assert reports["gsplat_nerfstudio_showcase"]["status"] == "ready_pending_execution"
    assert reports["mast3r_dust3r_vggetr_single_run"]["status"] == "ready_pending_method_command"
    assert reports["vggetr_candidate_unresolved"]["status"] in {
        "needs_identifier_confirmation",
        "unresolved",
        "configured slot",
    }
    audit = {row["method_id"]: row for row in inspected_payload["method_artifact_audit"]}
    assert audit["odm_offline_photogrammetry_baseline"]["status"] == "artifact_complete"
    assert audit["gsplat_nerfstudio_showcase"]["status"] == "artifact_complete"
    assert audit["vggetr_candidate_unresolved"]["status"] == "artifact_complete"
    assert any("optional report says dependency is missing" in action for action in inspected_payload["next_actions"])
    assert any("VGGeTR/VG2GT candidate slot needs a verified runner" in action for action in inspected_payload["next_actions"])

    inspected = runner.invoke(app, ["h100", "inspect-return", "--source", str(source)])
    assert inspected.exit_code == 0, inspected.output
    assert "optional odm_offline_photogrammetry_baseline: skipped_missing_dependency" in inspected.output
    assert "optional gsplat_nerfstudio_showcase: ready_pending_execution" in inspected.output

    direct_report_dir = tmp_path / "optional_direct"
    direct_report = runner.invoke(
        app,
        [
            "h100",
            "optional-report",
            "--source",
            str(source),
            "--output-dir",
            str(direct_report_dir),
        ],
    )
    assert direct_report.exit_code == 0, direct_report.output
    assert "H100 optional report done" in direct_report.output
    direct_payload = json.loads((direct_report_dir / "optional_method_report.json").read_text(encoding="utf-8"))
    assert direct_payload["optional_method_report_count"] == 6
    direct_markdown = (direct_report_dir / "OPTIONAL_METHOD_REPORT.md").read_text(encoding="utf-8")
    assert "Confirm the exact VGGeTR/VG2GT repository" in direct_markdown

    review_output = tmp_path / "reviews" / "h100_optional"
    imported = runner.invoke(
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
            "--dry-run",
        ],
    )
    assert imported.exit_code == 0, imported.output
    summary = json.loads((review_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["optional_method_report_count"] == 6
    assert summary["optional_method_status_counts"]["skipped_missing_dependency"] == 2
    assert summary["optional_method_status_counts"]["ready_pending_execution"] == 1
    assert (review_output / "expanded_artifacts" / "showcase_manifest.json").exists()
    optional_report = (review_output / "OPTIONAL_METHOD_REPORT.md").read_text(encoding="utf-8")
    assert "Optional Method Report" in optional_report
    assert "gsplat_nerfstudio_showcase" in optional_report
    assert "Confirm the exact VGGeTR/VG2GT repository" in optional_report
    next_steps = (review_output / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert "OPTIONAL_METHOD_REPORT.md" in next_steps
    html = (review_output / "index.html").read_text(encoding="utf-8")
    assert "Optional Method Manifests" in html
    assert "gsplat_nerfstudio_showcase" in html
    assert "vggetr_candidate_unresolved" in html

    launch = make_fake_launch_manifest(tmp_path)
    postflight_dir = tmp_path / "postflight_optional"
    postflight = verify_h100_return_against_launch(
        source=source,
        launch_manifest=launch,
        output_dir=postflight_dir,
    )
    assert len(postflight["optional_method_reports"]) == 6
    assert len(postflight["method_artifact_audit"]) == 9
    assert any("odm_offline_photogrammetry_baseline" in warning for warning in postflight["warnings"])
    markdown = (postflight_dir / "RETURN_POSTFLIGHT.md").read_text(encoding="utf-8")
    assert "Optional Method Reports" in markdown
    assert "Method Artifact Audit" in markdown
    assert "gsplat_nerfstudio_showcase" in markdown
def test_h100_inspect_return_rejects_corrupt_expanded_artifact(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="h100-corrupt-artifact-video",
        expanded_artifacts=True,
        corrupt_artifact=True,
    )
    report = validate_h100_return(source, extract_root=tmp_path / "corrupt_extract")

    assert any("colmap_sparse.zip is not a valid zip file" in issue for issue in report["issues"])



def test_h100_postflight_accepts_4090_hf_launch_manifest_schema(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="h100-postflight-4090-video",
        expanded_artifacts=True,
    )
    launch = make_fake_launch_manifest(tmp_path)
    data = json.loads(launch.read_text(encoding="utf-8"))
    data["schema_version"] = "runpod-4090-hf-launch-v1"
    data["package"]["sha256"] = "fake-4090-package-sha"
    launch.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    report = verify_h100_return_against_launch(
        source=source,
        launch_manifest=launch,
        output_dir=tmp_path / "postflight_4090",
    )

    assert report["status"] == "done_with_warnings"
    assert report["launch_package_sha256"] == "fake-4090-package-sha"
    assert report["missing_required_files"] == []
    assert report["missing_methods"] == []

def test_h100_postflight_return_compares_against_launch_manifest(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="h100-postflight-video",
        expanded_artifacts=True,
    )
    launch = make_fake_launch_manifest(tmp_path)
    output_dir = tmp_path / "postflight"

    report = verify_h100_return_against_launch(
        source=source,
        launch_manifest=launch,
        output_dir=output_dir,
    )

    assert report["status"] == "done_with_warnings"
    assert report["issues"] == []
    assert any("vggt_colmap_ba_windowed" in warning for warning in report["warnings"])
    assert "colmap_sparse.zip" in report["found_expanded_artifacts"]
    assert "evo_reports.zip" in report["found_expanded_artifacts"]
    assert report["missing_required_files"] == []
    assert report["missing_methods"] == []
    assert report["method_artifact_audit"][0]["method_id"] == "vggt_feedforward_full"
    assert (output_dir / "return_postflight.json").exists()
    assert (output_dir / "RETURN_POSTFLIGHT.md").exists()
    markdown = (output_dir / "RETURN_POSTFLIGHT.md").read_text(encoding="utf-8")
    assert "H100 Return Postflight" in markdown
    assert "done_with_warnings" in markdown
    assert "no geolocation" in markdown

    cli_result = runner.invoke(
        app,
        [
            "h100",
            "postflight-return",
            "--source",
            str(source),
            "--launch-manifest",
            str(launch),
            "--output-dir",
            str(tmp_path / "postflight_cli"),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "done_with_warnings" in cli_result.output


def test_h100_postflight_return_fails_on_missing_required_launch_file(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="h100-postflight-missing-file-video",
        expanded_artifacts=True,
    )
    launch = make_fake_launch_manifest(tmp_path, required_files=["manifest.json", "definitely_missing.json"])
    output_dir = tmp_path / "postflight_missing"

    report = verify_h100_return_against_launch(
        source=source,
        launch_manifest=launch,
        output_dir=output_dir,
    )

    assert report["status"] == "failed_soft"
    assert "definitely_missing.json" in report["missing_required_files"]
    assert any("definitely_missing.json" in issue for issue in report["issues"])
    assert (output_dir / "return_postflight.json").exists()



def test_runpod_4090_final_audit_return_writes_goal_boundary_artifacts(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="runpod-4090-final-audit-video",
        expanded_artifacts=True,
    )
    launch = make_fake_launch_manifest(tmp_path)
    data = json.loads(launch.read_text(encoding="utf-8"))
    data["schema_version"] = "runpod-4090-hf-launch-v1"
    data["package"]["sha256"] = "fake-4090-final-audit-sha"
    data["safety_warnings"] = [
        "no geolocation",
        "no map projection",
        "no meters",
        "no true speed/standoff/dive-angle claims",
        "no route/approach/launch/target-coordinate/guidance/next-maneuver inference",
        "relative VGGT frame",
        "local-only media",
    ]
    launch.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "final_audit"

    result = runner.invoke(
        app,
        [
            "runpod",
            "final-audit-4090-return",
            "--source",
            str(source),
            "--launch-manifest",
            str(launch),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RunPod 4090 final audit validated_with_warnings" in result.output
    audit = json.loads((output_dir / "runpod_4090_final_audit.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == "runpod-4090-final-audit-v1"
    assert audit["status"] == "validated_with_warnings"
    assert audit["postflight_status"] == "done_with_warnings"
    assert audit["launch_schema_version"] == "runpod-4090-hf-launch-v1"
    assert audit["launch_package_sha256"] == "fake-4090-final-audit-sha"
    assert audit["selected_bundle_count"] == 1
    assert audit["missing_required_files"] == []
    assert audit["missing_methods"] == []
    assert "no route/approach/launch/target-coordinate/guidance/next-maneuver inference" in audit["safety_warnings"]
    assert "does not add geolocation" in audit["completion_boundary"]
    markdown = (output_dir / "RUNPOD_4090_FINAL_AUDIT.md").read_text(encoding="utf-8")
    assert "RunPod 4090 Final Audit" in markdown
    assert "validated_with_warnings" in markdown
    assert "no geolocation" in markdown


def test_runpod_4090_final_audit_return_fails_closed_on_missing_required_file(tmp_path: Path):
    source = make_fake_h100_return(
        tmp_path,
        video_id="runpod-4090-final-audit-missing-file-video",
        expanded_artifacts=True,
    )
    launch = make_fake_launch_manifest(tmp_path, required_files=["manifest.json", "definitely_missing.json"])
    data = json.loads(launch.read_text(encoding="utf-8"))
    data["schema_version"] = "runpod-4090-hf-launch-v1"
    data["package"]["sha256"] = "fake-4090-final-audit-sha"
    data["safety_warnings"] = [
        "no geolocation",
        "no map projection",
        "no meters",
        "no true speed/standoff/dive-angle claims",
        "no route/approach/launch/target-coordinate/guidance/next-maneuver inference",
        "relative VGGT frame",
        "local-only media",
    ]
    launch.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "final_audit_missing"

    result = runner.invoke(
        app,
        [
            "runpod",
            "final-audit-4090-return",
            "--source",
            str(source),
            "--launch-manifest",
            str(launch),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "RunPod 4090 final audit failed_soft" in result.output
    audit = json.loads((output_dir / "runpod_4090_final_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "failed_soft"
    assert "definitely_missing.json" in audit["missing_required_files"]
    assert any("definitely_missing.json" in issue for issue in audit["issues"])
    markdown = (output_dir / "RUNPOD_4090_FINAL_AUDIT.md").read_text(encoding="utf-8")
    assert "failed_soft" in markdown
    assert "definitely_missing.json" in markdown
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



def test_h100_insights_command_builds_dashboard_from_review_run(tmp_path: Path):
    frame_manifests = []
    bundles = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"h100-insight-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    review_dir = tmp_path / "reviews" / "h100-full"
    run_three_clip_review(frame_manifests, bundles, review_dir, smooth=True)
    output_dir = review_dir / "insights"

    result = runner.invoke(
        app,
        [
            "h100",
            "insights",
            "--review-run",
            str(review_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "H100 insights done: 3 clips" in result.output
    payload = json.loads((output_dir / "h100_insights.json").read_text(encoding="utf-8"))
    assert payload["summary"]["clip_count"] == 3
    assert payload["summary"]["point_count"]["median"] > 0
    assert len(payload["clips"]) == 3
    html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "H100 VGGT Insight Review" in html
    assert "Search All Clip Metrics" in html
    assert "No geolocation" in html
    assert "file:///" not in html
    assert (output_dir / "H100_INSIGHTS.md").exists()
    assert (output_dir / "h100_clip_metrics.csv").exists()
    assert (output_dir / "verification_manifest.json").exists()
