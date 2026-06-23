from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

from .frames import read_frame_manifest
from .schemas import HeatmapLayerRecord, HeatmapManifest, model_to_dict

DEFAULT_LAYERS = (
    "frame_difference",
    "optical_flow",
    "blur",
    "visibility_change",
)
HEATMAP_WARNINGS = [
    "image-space diagnostics only",
    "no geolocation",
    "no meters",
    "local-only media",
]

LayerName = Literal[
    "frame_difference",
    "optical_flow",
    "blur",
    "visibility_change",
]


def generate_heatmaps_from_manifest(
    frame_manifest_path: Path,
    output_dir: Path,
    layers: Iterable[str] = DEFAULT_LAYERS,
) -> Path:
    requested_layers = tuple(layers)
    for layer_name in requested_layers:
        if layer_name not in DEFAULT_LAYERS:
            raise ValueError(f"unsupported heatmap layer: {layer_name}")

    frame_manifest_path = frame_manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_manifest = read_frame_manifest(frame_manifest_path)
    frames = [_read_bgr(frame.path) for frame in frame_manifest.frames]
    grays = [_to_gray(frame) for frame in frames]

    records: list[HeatmapLayerRecord] = []
    for sequence, frame_record in enumerate(frame_manifest.frames):
        layer_maps = _compute_layer_maps(grays, sequence)
        for layer_name in requested_layers:
            intensity = layer_maps[layer_name]
            heatmap = _colorize_alpha(intensity)
            layer_dir = output_dir / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = (
                layer_dir / f"frame_{sequence:04d}_{frame_record.frame_index:06d}.png"
            ).resolve()
            if not cv2.imwrite(str(heatmap_path), heatmap):
                raise RuntimeError(f"could not write heatmap: {heatmap_path}")

            value_min, value_max = _finite_min_max(intensity)
            records.append(
                HeatmapLayerRecord(
                    video_id=frame_record.video_id,
                    segment_id=frame_record.segment_id,
                    frame_index=frame_record.frame_index,
                    timestamp_sec=frame_record.timestamp_sec,
                    layer_name=layer_name,
                    layer_kind="image_space",
                    source_frame_path=frame_record.path,
                    heatmap_path=heatmap_path,
                    width=frame_record.width,
                    height=frame_record.height,
                    value_min=value_min,
                    value_max=value_max,
                    normalization="per-layer-frame min-max to uint8",
                    warnings=list(HEATMAP_WARNINGS),
                    local_only=True,
                )
            )

    manifest = HeatmapManifest(
        video_id=frame_manifest.video_id,
        segment_id=frame_manifest.segment_id,
        source_frame_manifest=frame_manifest_path,
        layers=records,
        warnings=list(HEATMAP_WARNINGS),
    )
    return write_heatmap_manifest(manifest, output_dir / "heatmaps.json")


def read_heatmap_manifest(path: Path) -> HeatmapManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return HeatmapManifest.model_validate(data)


def write_heatmap_manifest(manifest: HeatmapManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model_to_dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read frame image: {path}")
    return image


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _compute_layer_maps(
    grays: list[np.ndarray],
    index: int,
) -> dict[LayerName, np.ndarray]:
    current = grays[index]
    previous = grays[index - 1] if index > 0 else current
    next_frame = grays[index + 1] if index + 1 < len(grays) else current
    flow_reference = previous if index > 0 else next_frame

    frame_difference = cv2.absdiff(current, flow_reference).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        flow_reference,
        current,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    optical_flow = cv2.magnitude(flow[..., 0], flow[..., 1])
    blur = np.abs(cv2.Laplacian(current, cv2.CV_32F, ksize=3))
    visibility_change = np.maximum(
        cv2.absdiff(current, previous),
        cv2.absdiff(current, next_frame),
    ).astype(np.float32)

    return {
        "frame_difference": frame_difference,
        "optical_flow": optical_flow.astype(np.float32),
        "blur": blur.astype(np.float32),
        "visibility_change": visibility_change,
    }


def _colorize_alpha(values: np.ndarray) -> np.ndarray:
    alpha = _normalize_uint8(values)
    color = cv2.applyColorMap(alpha, cv2.COLORMAP_TURBO)
    return np.dstack([color, alpha])


def _normalize_uint8(values: np.ndarray) -> np.ndarray:
    value_min, value_max = _finite_min_max(values)
    if value_max <= value_min:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values.astype(np.float32) - value_min) / (value_max - value_min)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("heatmap values must include at least one finite value")
    return float(finite.min()), float(finite.max())
