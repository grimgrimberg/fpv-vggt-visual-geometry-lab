from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFETY_WARNINGS = [
    "offline historical-media reconstruction QA only",
    "no geolocation",
    "no map projection for real videos",
    "no meters / true speed / standoff claims",
    "no route, approach-corridor, launch, target-coordinate, guidance, or next-maneuver inference",
    "third-party media-derived artifacts are local-only by default",
]

REQUIRED_METHOD_IDS = [
    "vggt_feedforward_full",
    "vggt_colmap_ba_windowed",
    "colmap_classical_sift_sequential",
    "trajectory_preprocess_evo_eval",
    "openmvs_dense_mesh_baseline",
    "odm_offline_photogrammetry_baseline",
    "gsplat_nerfstudio_showcase",
    "mast3r_dust3r_vggetr_single_run",
    "vggetr_candidate_unresolved",
    "relative_depth_overlay_optional",
    "r3_relative_regression_optional",
    "lingbot_map_streaming_optional",
]


CORE_RETURN_FILES = [
    "manifest.json",
    "cloud_run.log",
    "environment.json",
    "expanded_runtime_check.json",
    "method_matrix.json",
    "return_contract.json",
    "method_stage_plan.json",
    "method_stage_report.json",
]

CORE_RETURN_REPORT_FILES = set(CORE_RETURN_FILES) | {
    "METHOD_CONTRACT.md",
    "cloud_summary.json",
    "job_manifest.json",
    "run.log",
    "selected_bundle_report.parquet",
    "selected_bundle_report.json",
    "tier_report.parquet",
    "tier_report.json",
    "quality_report.parquet",
    "quality_report.json",
    "hf_dataset_report.json",
    "hf_frame_pack_report.json",
}

EXPANDED_RETURN_ARTIFACTS = [
    "raw_predictions.zip",
    "colmap_sparse.zip",
    "converted_colmap_bundles.zip",
    "classical_colmap_sparse.zip",
    "classical_colmap_logs.zip",
    "registration_report.json",
    "points_ply.zip",
    "points_ply_report.json",
    "ba_logs.zip",
    "trajectories.zip",
    "evo_reports.zip",
    "trajectory_qc.parquet",
    "trajectory_qc.json",
    "odm_artifacts.zip",
    "odm_project_reports.zip",
    "odm_logs.zip",
    "odm_relative_artifacts_manifest.json",
    "openmvs_artifacts.zip",
    "openmvs_project_reports.zip",
    "openmvs_logs.zip",
    "openmvs_dense_mesh_manifest.json",
    "nerfstudio_gsplat_showcase.zip",
    "nerfstudio_projects.zip",
    "gsplat_exports.zip",
    "showcase_renders.zip",
    "showcase_manifest.json",
    "relative_depth_overlays.zip",
    "depth_overlay_manifest.json",
    "research_methods.zip",
    "research_methods_outputs.zip",
    "research_methods_manifest.json",
    "candidate_identity.md",
    "vggetr_feasibility_report.json",
    "r3_outputs.zip",
    "r3_training_report.json",
    "r3_dataset_prep_report.json",
    "r3_reconstruction_manifest.json",
    "lingbot_map_outputs.zip",
    "lingbot_map_manifest.json",
    "failure_packages.zip",
]


def expanded_return_artifact_files() -> list[str]:
    return list(EXPANDED_RETURN_ARTIFACTS)


def expanded_method_matrix() -> list[dict[str, Any]]:
    return [
        {
            "method_id": "vggt_feedforward_full",
            "status": "primary_production",
            "role": "primary dense geometry baseline and local review source",
            "input": "high_detail frame packs and selected stable-window packs, preferably 32-128 sampled frames",
            "output": "repo compact VGGT bundle, raw predictions.npz when available, RGB/confidence/depth point data, logs",
            "default_gpu": "RTX 4090 24GB for pilot; 48GB L40S/A6000/RTX6000 Ada for full dataset",
            "runtime_notes": "Inference-only. Do not describe as training or fine-tuning.",
            "risk": "pose jumps on cuts, blur, low texture, dynamic scenes, and edited footage; scale ambiguous",
            "safe_use": "relative reconstruction QA and visualization only",
            "return_contract": [
                "compact_bundles.zip",
                "raw_predictions.zip",
                "quality_report.parquet",
                "cloud_run.log",
            ],
        },
        {
            "method_id": "vggt_colmap_ba_windowed",
            "status": "primary_production_windowed",
            "role": "pose-stability upgrade for selected stable windows using official VGGT COLMAP export and BA",
            "input": "scene_dir/images for 16-64 frame stable windows",
            "output": "COLMAP sparse reconstruction, points.ply, converted repo bundle, BA logs",
            "default_gpu": "RTX 4090 24GB for 16-32 frames; 48GB+ for 64+ frames/high query points",
            "runtime_notes": "Run after VGGT feed-forward smoke test passes; CPU/RAM also matter.",
            "risk": "BA can fail if correspondences are weak; edited clips can break track consistency",
            "safe_use": "diagnostic comparison against raw VGGT; no metric, map, or route claims",
            "return_contract": [
                "colmap_sparse.zip",
                "converted_colmap_bundles.zip",
                "points_ply.zip",
                "ba_logs.zip",
            ],
        },
        {
            "method_id": "colmap_classical_sift_sequential",
            "status": "baseline_windowed",
            "role": "classical SfM sanity baseline independent of VGGT feed-forward poses",
            "input": "same stable-window images used by VGGT COLMAP BA",
            "output": "COLMAP sparse model, registered-frame count, track count, reprojection stats",
            "default_gpu": "GPU optional; CPU/RAM and disk IO usually matter more",
            "runtime_notes": "Use sequential matching/windowed images first. Failure is useful evidence on texture/blur limits.",
            "risk": "may fail on blur, low texture, compression artifacts, motion streaks, or scene dynamics",
            "safe_use": "reconstruction suitability baseline only",
            "return_contract": [
                "classical_colmap_sparse.zip",
                "classical_colmap_logs.zip",
                "registration_report.json",
            ],
        },
        {
            "method_id": "trajectory_preprocess_evo_eval",
            "status": "primary_diagnostic",
            "role": "trajectory normalization and relative trajectory comparison across successful methods",
            "input": "camera centers/quaternions from repo bundles and COLMAP outputs",
            "output": "TUM/KITTI-style trajectory exports, relative alignment reports, pose-jump metrics, method-vs-method plots",
            "default_gpu": "CPU only",
            "runtime_notes": "Use EVO for format handling and relative comparisons; no ground-truth claims without true ground truth.",
            "risk": "ATE/RPE terminology can imply truth; label as method consistency unless calibrated ground truth exists",
            "safe_use": "relative consistency diagnostics only; no physical speed, meters, or real trajectory claims",
            "return_contract": [
                "trajectories.zip",
                "evo_reports.zip",
                "trajectory_qc.parquet",
                "trajectory_qc.json",
            ],
        },
        {
            "method_id": "openmvs_dense_mesh_baseline",
            "status": "recommended_dense_baseline_windowed",
            "role": "COLMAP-to-OpenMVS dense point cloud, mesh, and texture attempt on selected successful windows",
            "input": "successful COLMAP/VGGT-COLMAP scene images plus sparse reconstruction, usually after COLMAP image_undistorter",
            "output": "OpenMVS scene.mvs, dense point cloud, mesh, optional refined/textured mesh, logs, and per-project report",
            "default_gpu": "CPU/RAM-heavy with optional CUDA acceleration depending on the OpenMVS build; use after VGGT/COLMAP succeeds",
            "runtime_notes": "Preferred ODM alternative for this workflow. Requires InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, and COLMAP image_undistorter for COLMAP sparse inputs.",
            "risk": "dense meshes can look convincing while sparse poses or undistortion are wrong; edited FPV footage and motion blur can produce holes or warped geometry",
            "safe_use": "offline relative dense-reconstruction baseline only; no geolocation, map projection, metric route, or operational interpretation",
            "return_contract": [
                "openmvs_artifacts.zip",
                "openmvs_project_reports.zip",
                "openmvs_logs.zip",
                "openmvs_dense_mesh_manifest.json",
            ],
        },
        {
            "method_id": "odm_offline_photogrammetry_baseline",
            "status": "optional_baseline_windowed",
            "role": "OpenDroneMap photogrammetry baseline on selected windows/subsets",
            "input": "short stable-window image folders with no GPS/geolocation metadata required",
            "output": "ODM reconstruction artifacts, logs, registered image summary, optional local visual artifact if produced without coordinates",
            "default_gpu": "CPU-heavy; 48GB RAM+ preferred. GPU is less central than VGGT.",
            "runtime_notes": "Run only on top-ranked windows first. Treat outputs as local relative reconstruction artifacts, not maps.",
            "risk": "ODM workflows are often built around aerial mapping assumptions; may produce misleading map-like artifacts if mislabeled",
            "safe_use": "offline relative photogrammetry baseline only; no geolocation, coordinates, map projection, or approach analysis",
            "return_contract": [
                "odm_project_reports.zip",
                "odm_logs.zip",
                "odm_relative_artifacts_manifest.json",
            ],
        },
        {
            "method_id": "gsplat_nerfstudio_showcase",
            "status": "optional_showcase_after_qc",
            "role": "high-quality 3D portfolio visualization from successful COLMAP/VGGT windows",
            "input": "COLMAP sparse output and images for top-ranked windows only",
            "output": "Gaussian splat / Nerfstudio viewer artifacts, preview renders, local-only showcase manifest",
            "default_gpu": "24GB can pilot one small scene; 48GB preferred for multiple scenes or higher quality",
            "runtime_notes": "Run only after reconstruction QC passes; do not run for all 161 clips first.",
            "risk": "can look impressive while hiding bad geometry; training/rendering can be time-consuming",
            "safe_use": "local visual showcase only; no map/geolocation/operational interpretation",
            "return_contract": [
                "nerfstudio_projects.zip",
                "gsplat_exports.zip",
                "showcase_renders.zip",
                "showcase_manifest.json",
            ],
        },
        {
            "method_id": "mast3r_dust3r_vggetr_single_run",
            "status": "single_run_if_configured",
            "role": "single-run secondary geometry baselines for visual comparison against VGGT and COLMAP",
            "input": "short stable windows only at first",
            "output": "relative poses/point clouds/reports when method runners are available",
            "default_gpu": "24GB pilot; 48GB preferred for full windows",
            "runtime_notes": "Executed in the same cloud run when method command templates or installed runners are available; skipped per method when missing.",
            "risk": "integration cost and model-specific coordinate conventions",
            "safe_use": "visual geometry benchmark, not physical trajectory claims",
            "return_contract": [
                "research_methods_outputs.zip",
                "research_methods_manifest.json",
            ],
        },
        {
            "method_id": "vggetr_candidate_unresolved",
            "status": "research_candidate_needs_identifier_confirmation",
            "role": "candidate transformer geometry baseline if the intended VGGeTR/VG^2GT/VG2GT implementation is confirmed",
            "input": "TBD after identifying the exact repository/checkpoint/API",
            "output": "TBD; method output must convert to repo compact bundle or trajectory export before review",
            "default_gpu": "TBD; assume 24GB pilot and 48GB production until proven otherwise",
            "runtime_notes": "Exact public runner/checkpoint/API is not bundled; configure VGGETR_COMMAND_TEMPLATE or VG2GT_COMMAND_TEMPLATE after confirming identity and license.",
            "risk": "wrong method identity, unstable code, incompatible licenses/checkpoints, or non-comparable outputs",
            "safe_use": "research-only configured slot; do not claim execution unless the method row is done, and do not block primary VGGT+COLMAP production on it",
            "return_contract": [
                "candidate_identity.md",
                "vggetr_feasibility_report.json",
            ],
        },
        {
            "method_id": "relative_depth_overlay_optional",
            "status": "optional_visual_diagnostic",
            "role": "make review pages visually richer and diagnose depth/texture failures",
            "input": "selected review frames",
            "output": "relative depth thumbnails/overlays and depth-consistency diagnostics",
            "default_gpu": "consumer GPU or T4 usually enough",
            "runtime_notes": "Use only for visual review overlays, not metric depth.",
            "risk": "monocular depth can look plausible while being wrong",
            "safe_use": "qualitative overlay only; no meters",
            "return_contract": [
                "relative_depth_overlays.zip",
                "depth_overlay_manifest.json",
            ],
        },
        {
            "method_id": "r3_relative_regression_optional",
            "status": "experimental_optional_if_installed",
            "role": "R3 relative-regression reconstruction/training smoke stage for long FPV frame windows",
            "input": "HF/downloaded frame packs or a prepared R3_DATA_ROOT in the upstream CUT3R-style supervised layout",
            "output": "R3 inference outputs, optional bounded training report, and local-only reconstruction manifest",
            "default_gpu": "RTX 4090 24GB for short inference/training smoke; 48GB+ for longer fine-tuning attempts",
            "runtime_notes": "Install from KevinXu02/R3 only when ENABLE_R3=1. Public R3 training expects prepared camera/depth supervision; do not pretend raw MP4 frames are supervised training data.",
            "risk": "external repo dependencies, non-commercial research restrictions in training components, and non-comparable coordinate conventions",
            "safe_use": "relative reconstruction benchmark only; no map, geolocation, metric route, or future maneuver claims",
            "return_contract": [
                "r3_outputs.zip",
                "r3_training_report.json",
                "r3_dataset_prep_report.json",
                "r3_reconstruction_manifest.json",
            ],
        },
        {
            "method_id": "lingbot_map_streaming_optional",
            "status": "experimental_optional_if_installed",
            "role": "LingBot-Map streaming reconstruction baseline for long image sequences",
            "input": "selected frame windows or HF-derived image folders, plus a configured LingBot checkpoint path",
            "output": "LingBot-Map local viewer/reconstruction outputs and manifest",
            "default_gpu": "RTX 4090 24GB pilot with short windows; 48GB+ if long sequences exceed memory",
            "runtime_notes": "Install from robbyant/lingbot-map only when ENABLE_LINGBOT_MAP=1. Prefer inference/reconstruction first; arbitrary FPV training is not enabled unless an upstream-compatible training recipe is supplied.",
            "risk": "streaming-memory configuration, external checkpoint licensing, and misleading map-like wording",
            "safe_use": "relative scene reconstruction only; no geolocation, map projection, operational route, or approach analysis",
            "return_contract": [
                "lingbot_map_outputs.zip",
                "lingbot_map_manifest.json",
            ],
        },
    ]


def expanded_return_contract() -> dict[str, Any]:
    return {
        "schema_version": "expanded-return-contract-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "safety_warnings": SAFETY_WARNINGS,
        "required_files": list(CORE_RETURN_FILES),
        "required_one_of": [
            ["h100_return.zip", "compact_bundles.zip", "selected/"],
            ["quality_report.json", "quality_report.parquet", "tier_report.json", "tier_report.parquet"],
        ],
        "optional_artifacts": expanded_return_artifact_files(),
        "forbidden_claims": [
            "geolocation",
            "map projection",
            "true speed",
            "standoff",
            "approach corridor",
            "target coordinates",
            "launch coordinates",
            "guidance",
            "next maneuver",
        ],
    }


def expanded_method_stage_plan(clips: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    clip_count = len(clips or [])
    return {
        "schema_version": "expanded-method-stage-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clip_count": clip_count,
        "safety_warnings": SAFETY_WARNINGS,
        "stages": [
            {
                "stage": "00_env_check",
                "method_id": "environment",
                "status": "implemented",
                "script": "scripts/00_env_check.py",
                "outputs": ["environment.json", "expanded_runtime_check.json"],
            },
            {
                "stage": "20_vggt_feedforward",
                "method_id": "vggt_feedforward_full",
                "status": "implemented",
                "script": "run_vggt_job.py",
                "outputs": ["bundles/", "predictions/", "bundles.zip", "cloud_summary.json"],
            },
            {
                "stage": "80_vggt_colmap_ba",
                "method_id": "vggt_colmap_ba_windowed",
                "status": "implemented_if_dependencies_present",
                "script": "scripts/80_run_expanded_methods.py",
                "dependency_contract": ["VGGT repository checkout with demo_colmap.py", "COLMAP runtime used by VGGT demo_colmap --use_ba"],
                "outputs": ["colmap_sparse.zip", "converted_colmap_bundles.zip", "points_ply.zip", "ba_logs.zip"],
                "command_template": "python demo_colmap.py --scene_dir=<scene_dir> --use_ba",
            },
            {
                "stage": "80_classical_colmap",
                "method_id": "colmap_classical_sift_sequential",
                "status": "implemented_if_dependencies_present",
                "script": "scripts/80_run_expanded_methods.py",
                "dependency_contract": ["colmap executable on PATH"],
                "outputs": ["classical_colmap_sparse.zip", "classical_colmap_logs.zip", "registration_report.json"],
                "command_template": "colmap feature_extractor -> sequential_matcher -> mapper",
            },
            {
                "stage": "80_trajectory_evo",
                "method_id": "trajectory_preprocess_evo_eval",
                "status": "implemented_for_relative_exports",
                "script": "scripts/80_run_expanded_methods.py",
                "dependency_contract": ["repo compact VGGT bundles"],
                "outputs": ["trajectories.zip", "evo_reports.zip", "trajectory_qc.parquet", "trajectory_qc.json"],
                "note": "Exports relative TUM-style trajectories. EVO metrics are method-consistency diagnostics only unless true ground truth is supplied.",
            },
            {
                "stage": "80_optional_visual_methods",
                "method_id": "openmvs_odm_gsplat_nerfstudio_and_research_methods",
                "status": "planned_optional_after_qc",
                "script": "scripts/80_run_expanded_methods.py",
                "dependency_contract": ["successful COLMAP/VGGT window QC first", "tool-specific runtime installed"],
                "outputs": ["openmvs_artifacts.zip", "odm_artifacts.zip", "nerfstudio_gsplat_showcase.zip", "relative_depth_overlays.zip", "research_methods.zip", "research_methods_outputs.zip"],
                "note": "Research methods are part of the single run when configured; missing tools produce per-method skipped reports instead of crashing the run.",
            },
        ],
    }


def method_contract_markdown() -> str:
    methods = expanded_method_matrix()
    lines = [
        "# Expanded Method Contract",
        "",
        "This contract keeps the next cloud run focused on offline relative reconstruction QA.",
        "",
        "## Safety",
        "",
    ]
    lines.extend(f"- {warning}" for warning in SAFETY_WARNINGS)
    lines.extend(["", "## Methods", ""])
    for method in methods:
        lines.extend(
            [
                f"### {method['method_id']}",
                "",
                f"- Status: `{method['status']}`",
                f"- Role: {method['role']}",
                f"- Safe use: {method['safe_use']}",
                f"- Return artifacts: {', '.join(method['return_contract'])}",
                "",
            ]
        )
    return "\n".join(lines)


def write_method_contract_files(output_dir: Path, clips: list[dict[str, Any]] | None = None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "method_matrix.json"
    contract_path = output_dir / "return_contract.json"
    readme_path = output_dir / "METHOD_CONTRACT.md"
    stage_plan_path = output_dir / "method_stage_plan.json"
    matrix_path.write_text(
        json.dumps(expanded_method_matrix(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    contract_path.write_text(
        json.dumps(expanded_return_contract(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme_path.write_text(method_contract_markdown(), encoding="utf-8")
    stage_plan_path.write_text(
        json.dumps(expanded_method_stage_plan(clips), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "method_matrix": str(matrix_path),
        "return_contract": str(contract_path),
        "method_contract": str(readme_path),
        "method_stage_plan": str(stage_plan_path),
    }


def validate_method_matrix(matrix_path: Path) -> dict[str, Any]:
    try:
        methods = json.loads(matrix_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"valid": False, "issues": [f"could not read method matrix: {exc}"]}
    return validate_method_rows(methods)


def validate_method_rows(methods: Any) -> dict[str, Any]:
    issues: list[str] = []
    if not isinstance(methods, list):
        return {"valid": False, "issues": ["method matrix must be a list"], "method_count": 0}
    ids = []
    for index, method in enumerate(methods):
        if not isinstance(method, dict):
            issues.append(f"method row {index} must be an object")
            continue
        method_id = method.get("method_id")
        if not method_id:
            issues.append(f"method row {index} missing method_id")
            continue
        ids.append(method_id)
        for required in ["status", "role", "safe_use", "return_contract"]:
            if not method.get(required):
                issues.append(f"{method_id} missing {required}")
        if not isinstance(method.get("return_contract"), list):
            issues.append(f"{method_id} return_contract must be a list")
        safe_use = str(method.get("safe_use", "")).lower()
        if method_id in {"openmvs_dense_mesh_baseline", "odm_offline_photogrammetry_baseline", "gsplat_nerfstudio_showcase"}:
            if "no " not in safe_use and "local" not in safe_use and "relative" not in safe_use:
                issues.append(f"{method_id} safe_use must label local/relative/non-geolocated scope")
    missing = sorted(set(REQUIRED_METHOD_IDS) - set(ids))
    extra = sorted(set(ids) - set(REQUIRED_METHOD_IDS))
    if missing:
        issues.append(f"missing required methods: {', '.join(missing)}")
    if len(ids) != len(set(ids)):
        issues.append("duplicate method_id values")
    return {
        "valid": not issues,
        "issues": issues,
        "method_count": len(ids),
        "method_ids": ids,
        "extra_method_ids": extra,
    }


@dataclass(frozen=True)
class ReturnSource:
    root: Path
    extracted_from: Path | None = None


def validate_return_layout(source: Path, extract_root: Path | None = None) -> dict[str, Any]:
    source = source.resolve()
    cleanup = False
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_to = (extract_root or source.parent / f"{source.stem}_contract_extract").resolve()
        if extract_to.exists():
            shutil.rmtree(extract_to)
        extract_to.mkdir(parents=True)
        with zipfile.ZipFile(source) as archive:
            bad = archive.testzip()
            if bad is not None:
                return {"valid": False, "issues": [f"bad zip member: {bad}"], "source": str(source)}
            try:
                _extract_zip_safely(archive, extract_to)
            except ValueError as exc:
                return {"valid": False, "issues": [str(exc)], "source": str(source)}
        return_source = ReturnSource(root=extract_to, extracted_from=source)
        cleanup = extract_root is None
    else:
        return_source = ReturnSource(root=source)
    try:
        return _validate_return_root(return_source)
    finally:
        if cleanup:
            shutil.rmtree(return_source.root, ignore_errors=True)


def _extract_zip_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"unsafe zip member path: {member.filename}")
    archive.extractall(destination)


def _validate_return_root(source: ReturnSource) -> dict[str, Any]:
    root = source.root
    issues: list[str] = []
    contract = expanded_return_contract()
    present = set()
    for path in root.rglob("*"):
        if path.is_file():
            present.add(path.relative_to(root).as_posix())
        elif path.is_dir():
            present.add(path.relative_to(root).as_posix() + "/")
    for required in contract["required_files"]:
        if required not in present:
            issues.append(f"missing required return file: {required}")
    for alternatives in contract["required_one_of"]:
        if not any(item in present or any(name.startswith(item.rstrip('/') + "/") for name in present) for item in alternatives):
            issues.append(f"missing one of: {', '.join(alternatives)}")
    matrix_path = root / "method_matrix.json"
    if matrix_path.exists():
        matrix_report = validate_method_matrix(matrix_path)
        if not matrix_report["valid"]:
            issues.extend(f"method_matrix: {issue}" for issue in matrix_report["issues"])
    return {
        "valid": not issues,
        "issues": issues,
        "source": str(source.extracted_from or root),
        "checked_root": str(root),
        "present_count": len(present),
    }
