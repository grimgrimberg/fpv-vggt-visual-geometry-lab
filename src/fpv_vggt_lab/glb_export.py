from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .vggt import load_bundle


GLTF_COMPONENT_FLOAT = 5126
GLTF_COMPONENT_UNSIGNED_BYTE = 5121
GLTF_MODE_POINTS = 0
GLTF_MODE_TRIANGLES = 4


@dataclass(frozen=True)
class GlbExportResult:
    output: Path
    metadata_output: Path
    point_count: int
    camera_count: int
    pose_jump_count: int
    confidence_threshold: float | None


def export_hf_style_glb(
    bundle_path: Path,
    output: Path,
    metadata_output: Path | None = None,
    confidence_percentile: float = 20.0,
    max_points: int = 300_000,
    show_cameras: bool = True,
) -> GlbExportResult:
    """Write a compact GLB scene inspired by the VGGT-omega demo renderer.

    The GLB contains a colored point primitive plus small camera-cone meshes. It
    is a local visual review artifact only; VGGT coordinates remain relative and
    scale ambiguous.
    """

    bundle = load_bundle(bundle_path)
    points = np.asarray(bundle.points, dtype=np.float32)
    colors = _point_colors(bundle, len(points))
    confidence = (
        np.asarray(bundle.point_confidence, dtype=np.float32)
        if bundle.point_confidence is not None
        else np.ones(len(points), dtype=np.float32)
    )

    selected, threshold = _selected_point_indices(confidence, confidence_percentile, max_points)
    selected_points = _scene_aligned_points(points[selected], bundle.camera_centers, bundle.quaternions_xyzw)
    selected_colors = colors[selected]

    builder = _GlbBuilder()
    builder.add_points(selected_points, selected_colors, name="VGGT RGB point cloud")

    jump_indices = _pose_jump_indices(bundle.camera_centers)
    camera_count = 0
    if show_cameras:
        aligned_centers = _scene_aligned_points(
            np.asarray(bundle.camera_centers, dtype=np.float32),
            bundle.camera_centers,
            bundle.quaternions_xyzw,
        )
        for index, (center, quaternion) in enumerate(zip(aligned_centers, bundle.quaternions_xyzw)):
            color = _camera_color(index, len(aligned_centers), index in jump_indices)
            builder.add_camera_cone(
                center=np.asarray(center, dtype=np.float32),
                quaternion=np.asarray(quaternion, dtype=np.float32),
                color=color,
                scale=_camera_scale(selected_points),
                name=f"camera_{index:03d}",
            )
            camera_count += 1

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(builder.to_glb())

    metadata_path = metadata_output.resolve() if metadata_output is not None else output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "label": "HF-style local GLB scene",
        "bundle": str(bundle_path.resolve()),
        "output": str(output),
        "point_count": int(len(selected_points)),
        "camera_count": int(camera_count),
        "pose_jump_count": int(len(jump_indices)),
        "pose_jump_indices": sorted(int(index) for index in jump_indices),
        "confidence_percentile": float(confidence_percentile),
        "confidence_threshold": threshold,
        "max_points": int(max_points),
        "coordinate_frame": "VGGT relative coordinate frame, first camera aligned for viewing",
        "warnings": [
            "local-only visual review artifact",
            "no geolocation",
            "no meters",
            "scale and global orientation are ambiguous",
            "camera cones are schematic visual glyphs, not aircraft body geometry",
            "red camera cones mark detected pose jumps",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return GlbExportResult(
        output=output,
        metadata_output=metadata_path,
        point_count=int(len(selected_points)),
        camera_count=int(camera_count),
        pose_jump_count=int(len(jump_indices)),
        confidence_threshold=threshold,
    )


def _point_colors(bundle: Any, point_count: int) -> np.ndarray:
    if bundle.point_colors_rgb is not None:
        colors = np.asarray(bundle.point_colors_rgb, dtype=np.uint8)
        if colors.shape == (point_count, 3):
            return colors
    if bundle.point_depth is not None:
        return _gradient_colors(np.asarray(bundle.point_depth, dtype=np.float32), invert=True)
    if bundle.point_confidence is not None:
        return _gradient_colors(np.asarray(bundle.point_confidence, dtype=np.float32), invert=False)
    return np.full((point_count, 3), 210, dtype=np.uint8)


def _gradient_colors(values: np.ndarray, invert: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full((len(values), 3), 210, dtype=np.uint8)
    lo, hi = np.percentile(values[finite], [5, 95])
    span = max(float(hi - lo), 1e-6)
    unit = np.clip((values - lo) / span, 0, 1)
    if invert:
        unit = 1.0 - unit
    colors = np.stack(
        [
            45 + 205 * unit,
            110 + 80 * (1.0 - np.abs(unit - 0.5) * 2.0),
            215 - 160 * unit,
        ],
        axis=1,
    )
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _selected_point_indices(
    confidence: np.ndarray,
    confidence_percentile: float,
    max_points: int,
) -> tuple[np.ndarray, float | None]:
    finite = np.isfinite(confidence)
    if not finite.any():
        indices = np.arange(len(confidence), dtype=np.int64)
        return _limit_indices(indices, max_points), None
    percentile = float(np.clip(confidence_percentile, 0, 100))
    threshold = float(np.percentile(confidence[finite], percentile))
    indices = np.flatnonzero(finite & (confidence >= threshold))
    if len(indices) == 0:
        indices = np.flatnonzero(finite)
    return _limit_indices(indices, max_points), threshold


def _limit_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(indices) <= max_points:
        return indices.astype(np.int64)
    chosen = np.linspace(0, len(indices) - 1, max_points).astype(np.int64)
    return indices[chosen].astype(np.int64)


def _scene_aligned_points(
    points: np.ndarray,
    camera_centers: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> np.ndarray:
    first_center = np.asarray(camera_centers[0], dtype=np.float32)
    first_quaternion = np.asarray(quaternions_xyzw[0], dtype=np.float32)
    rotation = _quat_to_matrix(first_quaternion).T
    opengl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    return ((points - first_center) @ rotation.T @ opengl.T).astype(np.float32)


def _camera_scale(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.05
    lower, upper = np.percentile(points, [5, 95], axis=0)
    span = float(np.linalg.norm(upper - lower))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    return max(span * 0.035, 0.02)


def _pose_jump_indices(camera_centers: np.ndarray) -> set[int]:
    centers = np.asarray(camera_centers, dtype=np.float32)
    if len(centers) < 3:
        return set()
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median = float(np.median(steps))
    if median <= 1e-9:
        return set()
    mad = float(np.median(np.abs(steps - median)))
    threshold = max(median * 3.0, median + 6.0 * max(mad, 1e-9))
    return {int(index + 1) for index, step in enumerate(steps) if float(step) > threshold}


def _camera_color(index: int, count: int, jump: bool) -> tuple[int, int, int, int]:
    if jump:
        return (230, 80, 70, 255)
    unit = index / max(count - 1, 1)
    return (
        int(75 + 120 * unit),
        int(190 - 90 * unit),
        int(235 - 155 * unit),
        255,
    )


class _GlbBuilder:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._buffer_views: list[dict[str, Any]] = []
        self._accessors: list[dict[str, Any]] = []
        self._meshes: list[dict[str, Any]] = []
        self._nodes: list[dict[str, Any]] = []

    def add_points(self, points: np.ndarray, colors: np.ndarray, name: str) -> None:
        positions_accessor = self._add_array(points.astype("<f4"), "VEC3", GLTF_COMPONENT_FLOAT)
        colors_accessor = self._add_array(colors.astype(np.uint8), "VEC3", GLTF_COMPONENT_UNSIGNED_BYTE, normalized=True)
        mesh_index = len(self._meshes)
        self._meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": positions_accessor,
                            "COLOR_0": colors_accessor,
                        },
                        "mode": GLTF_MODE_POINTS,
                    }
                ],
            }
        )
        self._nodes.append({"mesh": mesh_index, "name": name})

    def add_camera_cone(
        self,
        center: np.ndarray,
        quaternion: np.ndarray,
        color: tuple[int, int, int, int],
        scale: float,
        name: str,
    ) -> None:
        vertices, colors = _camera_cone_mesh(center, quaternion, color, scale)
        positions_accessor = self._add_array(vertices.astype("<f4"), "VEC3", GLTF_COMPONENT_FLOAT)
        colors_accessor = self._add_array(colors.astype(np.uint8), "VEC4", GLTF_COMPONENT_UNSIGNED_BYTE, normalized=True)
        mesh_index = len(self._meshes)
        self._meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": positions_accessor,
                            "COLOR_0": colors_accessor,
                        },
                        "mode": GLTF_MODE_TRIANGLES,
                    }
                ],
            }
        )
        self._nodes.append({"mesh": mesh_index, "name": name})

    def to_glb(self) -> bytes:
        self._pad_buffer(4)
        gltf = {
            "asset": {"version": "2.0", "generator": "fpv-vggt-visual-geometry-lab"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self._nodes)))}],
            "nodes": self._nodes,
            "meshes": self._meshes,
            "buffers": [{"byteLength": len(self._buffer)}],
            "bufferViews": self._buffer_views,
            "accessors": self._accessors,
        }
        json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
        bin_chunk = bytes(self._buffer)
        length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
        return (
            b"glTF"
            + struct.pack("<II", 2, length)
            + struct.pack("<I4s", len(json_chunk), b"JSON")
            + json_chunk
            + struct.pack("<I4s", len(bin_chunk), b"BIN\x00")
            + bin_chunk
        )

    def _add_array(
        self,
        array: np.ndarray,
        accessor_type: str,
        component_type: int,
        normalized: bool = False,
    ) -> int:
        self._pad_buffer(4)
        offset = len(self._buffer)
        data = np.ascontiguousarray(array)
        self._buffer.extend(data.tobytes())
        view_index = len(self._buffer_views)
        self._buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes})
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": int(data.shape[0]),
            "type": accessor_type,
        }
        if normalized:
            accessor["normalized"] = True
        if accessor_type == "VEC3" and component_type == GLTF_COMPONENT_FLOAT and len(data):
            accessor["min"] = data.min(axis=0).astype(float).tolist()
            accessor["max"] = data.max(axis=0).astype(float).tolist()
        self._accessors.append(accessor)
        return len(self._accessors) - 1

    def _pad_buffer(self, multiple: int) -> None:
        padding = (multiple - len(self._buffer) % multiple) % multiple
        if padding:
            self._buffer.extend(b"\x00" * padding)


def _camera_cone_mesh(
    center: np.ndarray,
    quaternion: np.ndarray,
    color: tuple[int, int, int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.55, -0.35, 1.4],
            [0.55, -0.35, 1.4],
            [0.55, 0.35, 1.4],
            [-0.55, 0.35, 1.4],
        ],
        dtype=np.float32,
    ) * float(scale)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 3],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    rotation = _quat_to_matrix(quaternion)
    opengl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    vertices = ((local @ opengl.T) @ rotation.T) + center
    triangle_vertices = vertices[faces.reshape(-1)].astype(np.float32)
    rgba = np.tile(np.asarray(color, dtype=np.uint8), (len(triangle_vertices), 1))
    return triangle_vertices, rgba


def _quat_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
