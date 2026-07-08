from __future__ import annotations

import json
import hashlib
import shutil
import zipfile
from pathlib import Path

from .frames import read_frame_manifest
from .schemas import model_to_dict


def create_cloud_job_package(frame_manifests: list[Path], output: Path) -> Path:
    if not frame_manifests:
        raise ValueError("at least one frame manifest is required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clips_root = output / "clips"
    if clips_root.exists():
        shutil.rmtree(clips_root)

    clips = []
    for frame_manifest_path in frame_manifests:
        manifest = read_frame_manifest(frame_manifest_path)
        clip_id = f"{manifest.video_id}__{manifest.segment_id}"
        clip_dir = clips_root / clip_id
        frames_dir = clip_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        updated_frames = []
        for ordinal, frame in enumerate(manifest.frames):
            source = Path(frame.path)
            destination = frames_dir / f"{ordinal:04d}_frame_{frame.frame_index:06d}{source.suffix}"
            shutil.copyfile(source, destination)
            updated_frames.append(frame.model_copy(update={"path": Path("frames") / destination.name}))
        packaged_manifest = manifest.model_copy(update={"frames": updated_frames})
        packaged_manifest_path = _write_packaged_frame_manifest(
            packaged_manifest, clip_dir / "frames.json"
        )
        clips.append(
            {
                "clip_id": clip_id,
                "video_id": manifest.video_id,
                "segment_id": manifest.segment_id,
                "frame_manifest": f"clips/{clip_id}/frames.json",
                "frame_count": len(packaged_manifest.frames),
                "frame_manifest_sha256": _sha256_file(packaged_manifest_path),
                "bundle_output": f"bundles/{manifest.video_id}/{manifest.segment_id}",
                "predictions_output": f"predictions/{manifest.video_id}/{manifest.segment_id}/predictions.npz",
            }
        )

    job_manifest = {
        "clips": clips,
        "warnings": [
            "local-only media-derived frames",
            "no geolocation",
            "no meters",
            "relative VGGT frame",
        ],
    }
    (output / "job_manifest.json").write_text(
        json.dumps(job_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "run_vggt_job.py").write_text(_runner_script(), encoding="utf-8")
    (output / "README.md").write_text(_readme(), encoding="utf-8")

    zip_path = output / "cloud_vggt_job.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _cloud_package_files(output):
            archive.write(path, arcname=path.relative_to(output))
    return output


def _cloud_package_files(output: Path) -> list[Path]:
    files: list[Path] = []
    for path in [output / "job_manifest.json", output / "run_vggt_job.py", output / "README.md"]:
        if path.is_file():
            files.append(path)
    clips_root = output / "clips"
    if clips_root.exists():
        files.extend(path for path in clips_root.rglob("*") if path.is_file())
    return files


def _write_packaged_frame_manifest(manifest, path: Path) -> Path:
    data = model_to_dict(manifest)
    data["source_video"] = "source_video_not_packaged"
    for frame in data["frames"]:
        frame["path"] = Path(frame["path"]).as_posix()
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runner_script() -> str:
    return r'''from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
RUN_LOG = ROOT / "cloud_run.log"
SUMMARY_PATH = ROOT / "cloud_summary.json"
BUNDLE_ARCHIVE = ROOT / "bundles.zip"


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str]) -> None:
    log("+ " + " ".join(command))
    subprocess.check_call(command)


def ensure_dependencies() -> None:
    try:
        import torch  # noqa: F401
        from vggt.models.vggt import VGGT  # noqa: F401
    except ModuleNotFoundError:
        run([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/vggt.git"])


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_manifest_provenance_errors(frame_manifest_path: Path, clip: dict, frame_manifest: dict) -> list[str]:
    errors = []
    expected_count = clip.get("frame_count")
    if expected_count is not None and len(frame_manifest.get("frames", [])) != expected_count:
        errors.append(
            f"frame_count mismatch: job_manifest={expected_count}, frames.json={len(frame_manifest.get('frames', []))}"
        )
    expected_sha = clip.get("frame_manifest_sha256")
    if expected_sha is not None:
        actual_sha = sha256_file(frame_manifest_path)
        if actual_sha != expected_sha:
            errors.append(
                f"frame_manifest_sha256 mismatch: job_manifest={expected_sha}, frames.json={actual_sha}"
            )
    return errors


def write_summary(summary: dict) -> None:
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def zip_bundles() -> str | None:
    bundles_dir = ROOT / "bundles"
    if not bundles_dir.exists():
        return None
    if BUNDLE_ARCHIVE.exists():
        BUNDLE_ARCHIVE.unlink()
    with zipfile.ZipFile(BUNDLE_ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundles_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(bundles_dir.parent))
    return str(BUNDLE_ARCHIVE)


def frame_paths(frame_manifest: dict, manifest_path: Path) -> list[str]:
    paths = []
    for frame in frame_manifest["frames"]:
        path = Path(frame["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths.append(str(path))
    return paths


def to_numpy_array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: to_numpy_array(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_numpy_array(item) for item in value]
    return value


def load_vggt_runtime():
    import torch
    from vggt.models.vggt import VGGT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this packaged VGGT job.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    started = perf_counter()
    log("load VGGT model")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    log(f"loaded VGGT model in {perf_counter() - started:.1f}s")
    return model, device, dtype


def run_vggt(frame_manifest_path: Path, predictions_output: Path, model, device: str, dtype) -> None:
    import torch
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    manifest = load_manifest(frame_manifest_path)
    started = perf_counter()
    images = load_and_preprocess_images(frame_paths(manifest, frame_manifest_path)).to(device)
    log(f"preprocessed {len(manifest['frames'])} frames in {perf_counter() - started:.1f}s")
    started = perf_counter()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(images)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            if "depth" in predictions:
                predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
                    to_numpy_array(predictions["depth"].squeeze(0)),
                    to_numpy_array(extrinsic.squeeze(0)),
                    to_numpy_array(intrinsic.squeeze(0)),
                )
    log(f"ran VGGT inference in {perf_counter() - started:.1f}s")
    started = perf_counter()
    arrays = {}
    for key, value in predictions.items():
        try:
            value = to_numpy_array(value)
            value = np.asarray(value)
        except Exception as exc:
            raise RuntimeError(f"could not convert prediction {key!r} to numpy: {exc}") from exc
        if value.dtype == object:
            log(f"skip non-array prediction {key}")
            continue
        if value.shape[:1] == (1,):
            value = value.squeeze(0)
        arrays[key] = value
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_output, **arrays)
    log(f"wrote predictions in {perf_counter() - started:.1f}s")

def camera_centers_and_quaternions(extrinsic: np.ndarray, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    matrices = extrinsic[:, :3, :] if extrinsic.shape == (frame_count, 4, 4) else extrinsic
    if matrices.shape != (frame_count, 3, 4):
        raise ValueError("extrinsic must have shape (frame_count, 3, 4) or (frame_count, 4, 4)")
    centers = []
    quats = []
    for matrix in matrices:
        r_wc = matrix[:3, :3]
        t = matrix[:3, 3]
        r_cw = r_wc.T
        centers.append(-r_cw @ t)
        quats.append(rotation_to_quaternion_xyzw(r_cw))
    return np.asarray(centers, dtype=np.float32), np.asarray(quats, dtype=np.float32)


def rotation_to_quaternion_xyzw(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s], dtype=np.float32)
    else:
        q = np.array([0, 0, 0, 1], dtype=np.float32)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 0 else np.array([0, 0, 0, 1], dtype=np.float32)


def mean_conf(pred: np.lib.npyio.NpzFile, frame_count: int) -> np.ndarray:
    for key in ["pose_conf", "camera_conf", "depth_conf", "point_conf"]:
        if key in pred and pred[key].shape[0] == frame_count:
            values = np.asarray(pred[key], dtype=np.float32)
            return values if values.ndim == 1 else values.reshape(frame_count, -1).mean(axis=1).astype(np.float32)
    return np.ones(frame_count, dtype=np.float32)


def point_bundle(
    pred: np.lib.npyio.NpzFile,
    frame_manifest_path: Path | None = None,
    max_points: int = 50000,
) -> dict | None:
    for key in ["world_points_from_depth", "point_map", "points", "world_points"]:
        if key not in pred:
            continue
        source = np.asarray(pred[key], dtype=np.float32)
        points = source.reshape(-1, 3) if source.ndim > 2 else source
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        finite = np.isfinite(points).all(axis=1)
        colors = point_colors(pred, len(points))
        if colors is None:
            colors = colors_from_frame_manifest(frame_manifest_path, source.shape, len(points))
        confidence = point_vector(pred, ["point_confidence", "point_conf", "depth_conf", "confidence"], len(points))
        depth = point_vector(pred, ["point_depth", "depth", "depth_map"], len(points))
        points = points[finite]
        if colors is not None:
            colors = colors[finite]
        if confidence is not None:
            confidence = confidence[finite]
        if depth is not None:
            depth = depth[finite]
        if len(points) > max_points:
            idx = np.linspace(0, len(points) - 1, max_points).round().astype(int)
            points = points[idx]
            if colors is not None:
                colors = colors[idx]
            if confidence is not None:
                confidence = confidence[idx]
            if depth is not None:
                depth = depth[idx]
        bundle = {"points": points.astype(np.float32)}
        if colors is not None:
            bundle["point_colors_rgb"] = colors
        if confidence is not None:
            bundle["point_confidence"] = confidence.astype(np.float32)
        if depth is not None:
            bundle["point_depth"] = depth.astype(np.float32)
        return bundle
    return None


def point_colors(pred: np.lib.npyio.NpzFile, point_count: int) -> np.ndarray | None:
    for key in ["point_colors_rgb", "colors_rgb", "rgb", "images", "image"]:
        if key not in pred:
            continue
        values = np.asarray(pred[key])
        if values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])
        if values.ndim != 2 or values.shape != (point_count, 3):
            continue
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            continue
        if values.size and float(values.max()) <= 1.0:
            values = values * 255.0
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return None


def colors_from_frame_manifest(
    frame_manifest_path: Path | None,
    point_source_shape: tuple,
    point_count: int,
) -> np.ndarray | None:
    if frame_manifest_path is None or len(point_source_shape) < 4:
        return None
    frame_count, height, width = [int(value) for value in point_source_shape[:3]]
    if frame_count <= 0 or height <= 0 or width <= 0:
        return None
    try:
        manifest = load_manifest(frame_manifest_path)
        rows = manifest.get("frames", [])[:frame_count]
    except Exception:
        return None
    if len(rows) != frame_count:
        return None
    colors = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = frame_manifest_path.parent / path
        image = read_rgb_image(path, width=width, height=height)
        if image is None:
            return None
        colors.append(image.reshape(-1, 3))
    values = np.concatenate(colors, axis=0)
    if values.shape != (point_count, 3):
        return None
    return values.astype(np.uint8)


def read_rgb_image(path: Path, width: int, height: int) -> np.ndarray | None:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image.astype(np.uint8)
    except Exception:
        try:
            from PIL import Image

            with Image.open(path) as image_handle:
                image = image_handle.convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height))
                return np.asarray(image, dtype=np.uint8)
        except Exception:
            return None


def point_vector(pred: np.lib.npyio.NpzFile, names: list[str], point_count: int) -> np.ndarray | None:
    for key in names:
        if key not in pred:
            continue
        values = np.asarray(pred[key], dtype=np.float32).reshape(-1)
        if values.shape != (point_count,):
            continue
        if not np.isfinite(values).all():
            continue
        return values.astype(np.float32)
    return None


def normalize_bundle(frame_manifest_path: Path, predictions_path: Path, bundle_output: Path) -> None:
    manifest = load_manifest(frame_manifest_path)
    pred = np.load(predictions_path, allow_pickle=False)
    frame_count = len(manifest["frames"])
    centers, quats = camera_centers_and_quaternions(np.asarray(pred["extrinsic"], dtype=np.float32), frame_count)
    bundle_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_output / "cameras.npz",
        camera_centers=centers,
        quaternions_xyzw=quats,
        valid_pose_mask=np.isfinite(centers).all(axis=1),
        pose_confidence=mean_conf(pred, frame_count),
    )
    points = point_bundle(pred, frame_manifest_path)
    if points is not None:
        np.savez_compressed(bundle_output / "points.npz", **points)
    metadata = {
        "schema_version": "v1",
        "video_id": manifest["video_id"],
        "segment_id": manifest["segment_id"],
        "source_tool": "local-vggt",
        "source_url_or_repo": "https://github.com/facebookresearch/vggt",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame_indices": [row["frame_index"] for row in manifest["frames"]],
        "frame_timestamps_sec": [row["timestamp_sec"] for row in manifest["frames"]],
        "warnings": [
            "Generated in cloud/GPU job; coordinates remain relative and scale ambiguous.",
            "No geolocation, no meters, no physical speed/standoff/dive-angle claims.",
        ],
        "coordinate_frame": "VGGT relative coordinate frame",
    }
    (bundle_output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def validate_normalized_bundle(bundle_output: Path, frame_count: int) -> list[str]:
    errors = []
    metadata_path = bundle_output / "metadata.json"
    cameras_path = bundle_output / "cameras.npz"
    if not metadata_path.exists():
        errors.append("missing metadata.json")
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("coordinate_frame", "").lower().find("relative") == -1:
                errors.append("coordinate_frame does not state relative coordinates")
            frame_indices = metadata.get("frame_indices", [])
            frame_timestamps = metadata.get("frame_timestamps_sec", [])
            if len(frame_indices) != frame_count:
                errors.append("metadata frame_indices length mismatch")
            if len(frame_timestamps) != len(frame_indices):
                errors.append("metadata frame_timestamps_sec length mismatch")
            if any(index < 0 for index in frame_indices):
                errors.append("metadata frame_indices must be non-negative")
            if any(next_index <= index for index, next_index in zip(frame_indices, frame_indices[1:])):
                errors.append("metadata frame_indices must be strictly increasing")
            if any(timestamp < 0 for timestamp in frame_timestamps):
                errors.append("metadata frame_timestamps_sec must be non-negative")
            if any(next_timestamp < timestamp for timestamp, next_timestamp in zip(frame_timestamps, frame_timestamps[1:])):
                errors.append("metadata frame_timestamps_sec must be non-decreasing")
        except Exception as exc:
            errors.append(f"invalid metadata.json: {exc}")
    if not cameras_path.exists():
        errors.append("missing cameras.npz")
    else:
        try:
            cameras = np.load(cameras_path, allow_pickle=False)
            expected_shapes = {
                "camera_centers": (frame_count, 3),
                "quaternions_xyzw": (frame_count, 4),
                "valid_pose_mask": (frame_count,),
                "pose_confidence": (frame_count,),
            }
            for key, shape in expected_shapes.items():
                if key not in cameras:
                    errors.append(f"missing {key} in cameras.npz")
                elif cameras[key].shape != shape:
                    errors.append(f"{key} has shape {cameras[key].shape}, expected {shape}")
            for key in ["camera_centers", "quaternions_xyzw", "pose_confidence"]:
                if key in cameras and not np.isfinite(cameras[key]).all():
                    errors.append(f"{key} contains non-finite values")
        except Exception as exc:
            errors.append(f"invalid cameras.npz: {exc}")
    points_path = bundle_output / "points.npz"
    if points_path.exists():
        try:
            points_data = np.load(points_path, allow_pickle=False)
            if "points" not in points_data:
                errors.append("missing points in points.npz")
            elif points_data["points"].ndim != 2 or points_data["points"].shape[1] != 3:
                errors.append("points must have shape (point_count, 3)")
            elif not np.isfinite(points_data["points"]).all():
                errors.append("points contains non-finite values")
            else:
                point_count = points_data["points"].shape[0]
                if "point_colors_rgb" in points_data:
                    colors = points_data["point_colors_rgb"]
                    if colors.shape != (point_count, 3):
                        errors.append("point_colors_rgb must have shape (point_count, 3)")
                    elif not np.isfinite(colors).all():
                        errors.append("point_colors_rgb contains non-finite values")
                for key in ["point_confidence", "point_depth"]:
                    if key not in points_data:
                        continue
                    values = points_data[key]
                    if values.shape != (point_count,):
                        errors.append(f"{key} must have shape (point_count,)")
                    elif not np.isfinite(values).all():
                        errors.append(f"{key} contains non-finite values")
        except Exception as exc:
            errors.append(f"invalid points.npz: {exc}")
    return errors


def existing_valid_bundle(bundle_output: Path, frame_count: int) -> tuple[bool, list[str]]:
    if not bundle_output.exists():
        return False, []
    validation_errors = validate_normalized_bundle(bundle_output, frame_count)
    return not validation_errors, validation_errors



def cleanup_gpu() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        log(f"gpu cleanup warning: {exc}")

def main() -> None:
    if RUN_LOG.exists():
        RUN_LOG.unlink()
    manifest = json.loads((ROOT / "job_manifest.json").read_text(encoding="utf-8"))
    summary = {
        "status": "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "warnings": manifest.get("warnings", []) + [
            "no speed/standoff/dive-angle claims",
            "local-only media-derived frames",
        ],
        "clips": [],
        "artifacts": {
            "run_log": str(RUN_LOG),
            "summary": str(SUMMARY_PATH),
            "bundles_zip": None,
        },
    }
    log("This job is for offline VGGT reconstruction only.")
    log("No geolocation, no meters, no speed/standoff/dive-angle claims.")
    try:
        ensure_dependencies()
    except Exception as exc:
        summary["dependency_error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_summary(summary)
        raise

    model_runtime = None
    for clip in manifest["clips"]:
        frame_manifest = ROOT / clip["frame_manifest"]
        predictions_output = ROOT / clip["predictions_output"]
        bundle_output = ROOT / clip["bundle_output"]
        clip_summary = {
            "clip_id": clip["clip_id"],
            "video_id": clip["video_id"],
            "segment_id": clip["segment_id"],
            "frame_manifest": str(frame_manifest),
            "predictions_output": str(predictions_output),
            "bundle_output": str(bundle_output),
            "status": "failed",
            "bundle_valid": False,
            "skipped_existing": False,
            "validation_errors": [],
            "existing_validation_errors": [],
            "error": None,
        }
        try:
            log(f"start clip {clip['clip_id']}")
            frame_manifest_data = load_manifest(frame_manifest)
            provenance_errors = frame_manifest_provenance_errors(
                frame_manifest, clip, frame_manifest_data
            )
            if provenance_errors:
                raise ValueError("; ".join(provenance_errors))
            frame_count = len(frame_manifest_data["frames"])
            already_valid, existing_errors = existing_valid_bundle(bundle_output, frame_count)
            if already_valid:
                clip_summary["bundle_valid"] = True
                clip_summary["status"] = "done"
                clip_summary["skipped_existing"] = True
                log(f"skip existing valid bundle for clip {clip['clip_id']}")
                summary["clips"].append(clip_summary)
                continue
            if existing_errors:
                clip_summary["existing_validation_errors"] = existing_errors
                log(
                    f"existing bundle invalid for clip {clip['clip_id']}; rerunning VGGT: "
                    + "; ".join(existing_errors)
                )
            if model_runtime is None:
                model_runtime = load_vggt_runtime()
            run_vggt(frame_manifest, predictions_output, *model_runtime)
            normalize_bundle(frame_manifest, predictions_output, bundle_output)
            validation_errors = validate_normalized_bundle(bundle_output, frame_count)
            clip_summary["validation_errors"] = validation_errors
            if validation_errors:
                raise ValueError("bundle validation failed: " + "; ".join(validation_errors))
            clip_summary["bundle_valid"] = True
            clip_summary["status"] = "done"
            log(f"done clip {clip['clip_id']}")
        except Exception as exc:
            clip_summary["error"] = str(exc)
            clip_summary["traceback"] = traceback.format_exc()
            log(f"failed clip {clip['clip_id']}: {exc}")
        finally:
            cleanup_gpu()
        summary["clips"].append(clip_summary)

    summary["artifacts"]["bundles_zip"] = zip_bundles()
    if summary["clips"] and all(clip["status"] == "done" for clip in summary["clips"]):
        summary["status"] = "done"
    elif any(clip["status"] == "done" for clip in summary["clips"]):
        summary["status"] = "done_partial"
    else:
        summary["status"] = "failed"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(summary)
    log("Done. Return bundles.zip, cloud_summary.json, and cloud_run.log to the main workstation.")
    if summary["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


def _readme() -> str:
    return """# Cloud VGGT Job

This package contains sampled frames and frame manifests for offline VGGT
reconstruction. It is local-only media-derived material.

Run in a CUDA GPU environment:

```bash
python run_vggt_job.py
```

The script installs `facebookresearch/vggt` if it is missing, runs VGGT for each
clip listed in `job_manifest.json`, writes normalized repo-owned bundles, and
validates each bundle before marking a clip done. Reruns skip clips that already
have a valid normalized bundle, so interrupted jobs can resume without
recomputing completed clips.

`job_manifest.json` records `frame_count` and `frame_manifest_sha256` for each
packaged clip. Use those fields only for handoff/debugging provenance.
It also writes:

- `cloud_run.log`
- `cloud_summary.json`
- `bundles.zip`

Return `bundles.zip`, `cloud_summary.json`, and `cloud_run.log` to the main
workstation, then import and validate `bundles.zip` directly:

```bash
fpv vggt import-cloud-job --source bundles.zip --output-root data/vggt --report outputs/reviews/cloud_bundle_import.json
```

After import, run `fpv review audit-three` and `fpv review run-three`.

Safety boundaries: no geolocation, no meters, no real speed/standoff/dive-angle
claims, no route/guidance/targeting analysis.
"""
