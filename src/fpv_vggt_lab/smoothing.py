from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schemas import PoseSmoothingMetadata, model_to_dict
from .vggt import load_bundle


def smooth_relative_poses(
    bundle_path: Path,
    summary_path: Path,
    output: Path,
    metadata_output: Path | None = None,
    window: int = 3,
) -> tuple[Path, Path | None]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reliability = summary.get("reliability", {})
    if not reliability.get("safe_to_use_for_descriptors", False):
        raise ValueError("reconstruction is not safe for relative pose smoothing")

    bundle = load_bundle(bundle_path)
    centers = bundle.camera_centers
    smoothed = moving_average(centers, window=window)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        smoothed_camera_centers=smoothed,
        original_camera_centers=centers,
        quaternions_xyzw=bundle.quaternions_xyzw,
    )

    meta_path = metadata_output
    if meta_path is not None:
        metadata = PoseSmoothingMetadata(
            video_id=bundle.metadata.video_id,
            segment_id=bundle.metadata.segment_id,
            warnings=[
                "experimental",
                "scale ambiguous",
                "not physically calibrated",
                "not guidance, control, or prediction",
            ],
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(model_to_dict(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return output, meta_path


def moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.copy()
    radius = window // 2
    smoothed = np.zeros_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed[index] = values[start:end].mean(axis=0)
    return smoothed
