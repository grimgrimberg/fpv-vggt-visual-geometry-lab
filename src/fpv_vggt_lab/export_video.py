from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .frames import read_frame_manifest
from .vggt import VggtBundle, load_bundle


def export_side_by_side_mp4(
    frame_manifest_path: Path,
    bundle_path: Path,
    summary_path: Path,
    output: Path,
    fps: float = 8.0,
    width: int = 1280,
    height: int = 720,
    point_budget: int = 8000,
    confidence_min: float = 0.0,
    relative_depth_quantile: float = 1.0,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if width < 320 or height < 240:
        raise ValueError("width and height must be at least 320x240")
    if point_budget <= 0:
        raise ValueError("point_budget must be positive")

    manifest = read_frame_manifest(frame_manifest_path)
    bundle = load_bundle(bundle_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer: {output}")

    right_width = width // 2
    left_width = width - right_width
    point_rows = _selected_point_rows(
        bundle=bundle,
        point_budget=point_budget,
        confidence_min=confidence_min,
        relative_depth_quantile=relative_depth_quantile,
    )
    try:
        for index, frame in enumerate(manifest.frames):
            image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read sampled frame: {frame.path}")
            left = _letterbox(image, left_width, height)
            right = _geometry_panel(
                bundle=bundle,
                active_index=index,
                point_rows=point_rows,
                width=right_width,
                height=height,
                summary=summary,
            )
            _write_text(
                left,
                f"FPV sample {index + 1}/{len(manifest.frames)} | frame {frame.frame_index} | {frame.timestamp_sec:.3f}s",
                (12, 26),
            )
            composed = np.concatenate([left, right], axis=1)
            writer.write(composed)
    finally:
        writer.release()
    return output


def _selected_point_rows(
    bundle: VggtBundle,
    point_budget: int,
    confidence_min: float,
    relative_depth_quantile: float,
) -> list[dict[str, Any]]:
    if bundle.points is None or len(bundle.points) == 0:
        return []
    points = np.asarray(bundle.points, dtype=np.float32)
    confidence = (
        np.asarray(bundle.point_confidence, dtype=np.float32)
        if bundle.point_confidence is not None
        else np.ones(len(points), dtype=np.float32)
    )
    depth = _point_depth(bundle)
    colors = bundle.point_colors_rgb

    mask = np.isfinite(points).all(axis=1)
    if confidence_min > 0:
        mask &= confidence >= float(confidence_min)
    if relative_depth_quantile < 1.0 and len(depth):
        masked_depth = depth[mask]
        if len(masked_depth):
            limit = float(np.quantile(masked_depth, np.clip(relative_depth_quantile, 0.0, 1.0)))
            mask &= depth <= limit
    indices = np.flatnonzero(mask)
    if len(indices) > point_budget:
        indices = indices[np.linspace(0, len(indices) - 1, point_budget).round().astype(int)]

    rows = []
    max_depth = float(depth[indices].max()) if len(indices) else 1.0
    for index in indices:
        color = None
        if colors is not None:
            rgb = np.asarray(colors[index], dtype=np.uint8)
            color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        else:
            color = _heat_color(1.0 - float(depth[index]) / max(max_depth, 1e-6))
        rows.append(
            {
                "point": points[index],
                "color": color,
                "confidence": float(confidence[index]),
                "depth": float(depth[index]),
            }
        )
    return rows


def _point_depth(bundle: VggtBundle) -> np.ndarray:
    if bundle.points is None:
        return np.asarray([], dtype=np.float32)
    if bundle.point_depth is not None:
        return np.asarray(bundle.point_depth, dtype=np.float32)
    origin = bundle.camera_centers[0] if len(bundle.camera_centers) else bundle.points.mean(axis=0)
    return np.linalg.norm(bundle.points - origin[None, :], axis=1).astype(np.float32)


def _geometry_panel(
    bundle: VggtBundle,
    active_index: int,
    point_rows: list[dict[str, Any]],
    width: int,
    height: int,
    summary: dict[str, Any],
) -> np.ndarray:
    canvas = np.full((height, width, 3), 17, dtype=np.uint8)
    path = np.asarray(bundle.camera_centers, dtype=np.float32)
    scene_points = [row["point"] for row in point_rows] + [point for point in path]
    scene = _scene(scene_points, width, height)

    for row in point_rows:
        x, y = _project(row["point"], scene)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 1, row["color"], thickness=-1, lineType=cv2.LINE_AA)

    projected_path = [_project(point, scene) for point in path]
    if len(projected_path) > 1:
        cv2.polylines(
            canvas,
            [np.asarray(projected_path, dtype=np.int32)],
            isClosed=False,
            color=(255, 194, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    stride = max(1, int(np.ceil(len(path) / 12)))
    for index in range(0, len(path), stride):
        _draw_frustum(canvas, path[index], bundle.quaternions_xyzw[index], scene, (120, 120, 120), 1)
    if active_index < len(path):
        _draw_frustum(
            canvas,
            path[active_index],
            bundle.quaternions_xyzw[active_index],
            scene,
            (55, 235, 255),
            2,
        )
        _draw_pose_axes(canvas, path[active_index], bundle.quaternions_xyzw[active_index], scene)

    reliability = summary.get("reliability", {})
    _write_text(canvas, "Relative VGGT frame | no meters | no geolocation", (12, 26))
    _write_text(
        canvas,
        f"reliability: {reliability.get('label', 'unknown')} | score: {reliability.get('score', 'n/a')}",
        (12, 50),
    )
    return canvas


def _scene(points: list[np.ndarray], width: int, height: int) -> dict[str, Any]:
    if not points:
        points = [np.zeros(3, dtype=np.float32)]
    values = np.asarray(points, dtype=np.float32)
    center = values.mean(axis=0)
    ranges = np.ptp(values, axis=0)
    span = float(max(ranges.max(), 1e-6))
    return {
        "center": center,
        "scale": min(width, height) * 0.72 / span,
        "width": width,
        "height": height,
        "yaw": -0.65,
        "pitch": -0.45,
    }


def _project(point: np.ndarray, scene: dict[str, Any]) -> tuple[int, int]:
    value = np.asarray(point, dtype=np.float32) - scene["center"]
    yaw = scene["yaw"]
    pitch = scene["pitch"]
    cy = float(np.cos(yaw))
    sy = float(np.sin(yaw))
    cp = float(np.cos(pitch))
    sp = float(np.sin(pitch))
    x = cy * value[0] + sy * value[2]
    z = -sy * value[0] + cy * value[2]
    y = cp * value[1] - sp * z
    scale = scene["scale"]
    return (
        int(round(scene["width"] / 2 + x * scale)),
        int(round(scene["height"] / 2 - y * scale)),
    )


def _draw_pose_axes(
    canvas: np.ndarray,
    origin: np.ndarray,
    quaternion: np.ndarray,
    scene: dict[str, Any],
) -> None:
    axis_scale = 0.08 * max(1.0 / scene["scale"] * min(scene["width"], scene["height"]), 1e-3)
    axes = [
        (np.array([1, 0, 0], dtype=np.float32), (60, 90, 255)),
        (np.array([0, 1, 0], dtype=np.float32), (90, 210, 105)),
        (np.array([0, 0, 1], dtype=np.float32), (255, 165, 90)),
    ]
    for axis, color in axes:
        end = origin + _rotate_by_quaternion(axis, quaternion) * axis_scale
        cv2.line(canvas, _project(origin, scene), _project(end, scene), color, 2, cv2.LINE_AA)
    cv2.circle(canvas, _project(origin, scene), 5, (55, 235, 255), -1, cv2.LINE_AA)


def _draw_frustum(
    canvas: np.ndarray,
    origin: np.ndarray,
    quaternion: np.ndarray,
    scene: dict[str, Any],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    size = 0.055 * max(1.0 / scene["scale"] * min(scene["width"], scene["height"]), 1e-3)
    forward = _rotate_by_quaternion(np.array([0, 0, 1], dtype=np.float32), quaternion)
    right = _rotate_by_quaternion(np.array([1, 0, 0], dtype=np.float32), quaternion)
    up = _rotate_by_quaternion(np.array([0, 1, 0], dtype=np.float32), quaternion)
    center = origin + forward * size * 2.2
    corners = [
        center + right * size + up * size * 0.65,
        center - right * size + up * size * 0.65,
        center - right * size - up * size * 0.65,
        center + right * size - up * size * 0.65,
    ]
    for corner in corners:
        cv2.line(canvas, _project(origin, scene), _project(corner, scene), color, thickness, cv2.LINE_AA)
    for index, corner in enumerate(corners):
        cv2.line(
            canvas,
            _project(corner, scene),
            _project(corners[(index + 1) % len(corners)], scene),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _rotate_by_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float32)
    uv = np.array(
        [
            y * vector[2] - z * vector[1],
            z * vector[0] - x * vector[2],
            x * vector[1] - y * vector[0],
        ],
        dtype=np.float32,
    )
    uuv = np.array(
        [
            y * uv[2] - z * uv[1],
            z * uv[0] - x * uv[2],
            x * uv[1] - y * uv[0],
        ],
        dtype=np.float32,
    )
    return vector + 2 * (w * uv + uuv)


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    output = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    x = (width - new_size[0]) // 2
    y = (height - new_size[1]) // 2
    output[y : y + new_size[1], x : x + new_size[0]] = resized
    return output


def _write_text(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _heat_color(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    return (
        int(70 + 140 * (1.0 - value)),
        int(90 + 140 * value),
        int(210 - 140 * value),
    )
