from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


class FrameQuality(BaseModel):
    blur_score: float = Field(ge=0)
    brightness_mean: float = Field(ge=0, le=255)
    contrast_std: float = Field(ge=0)


class FrameRecord(BaseModel):
    video_id: str
    segment_id: str
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    resized_long_edge: int | None = Field(default=None, ge=1)
    quality: FrameQuality


class FrameManifest(BaseModel):
    video_id: str
    segment_id: str
    source_video: Path
    frame_count: int = Field(gt=0)
    source_fps: float = Field(gt=0)
    frames: list[FrameRecord]

    @field_validator("frames")
    @classmethod
    def frames_must_not_be_empty(cls, value: list[FrameRecord]) -> list[FrameRecord]:
        if not value:
            raise ValueError("frame manifest must include at least one frame")
        return value


class HeatmapLayerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    segment_id: str
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    layer_name: Literal[
        "frame_difference",
        "optical_flow",
        "blur",
        "visibility_change",
    ]
    layer_kind: Literal["image_space"] = "image_space"
    source_frame_path: Path
    heatmap_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    value_min: float = Field(allow_inf_nan=False)
    value_max: float = Field(allow_inf_nan=False)
    normalization: str = "per-layer-frame min-max to uint8"
    warnings: list[str] = Field(default_factory=list)
    local_only: bool = True

    @field_validator("local_only")
    @classmethod
    def must_be_local_only(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("heatmap layers from real clips must be local-only")
        return value

    @model_validator(mode="after")
    def value_range_must_be_ordered(self) -> HeatmapLayerRecord:
        if self.value_min > self.value_max:
            raise ValueError("value_min must be less than or equal to value_max")
        return self


class HeatmapManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    segment_id: str
    source_frame_manifest: Path
    layers: list[HeatmapLayerRecord]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("layers")
    @classmethod
    def layers_must_not_be_empty(
        cls, value: list[HeatmapLayerRecord]
    ) -> list[HeatmapLayerRecord]:
        if not value:
            raise ValueError("heatmap manifest must include at least one layer")
        return value

    @model_validator(mode="after")
    def layers_must_match_manifest(self) -> HeatmapManifest:
        for layer in self.layers:
            if layer.video_id != self.video_id or layer.segment_id != self.segment_id:
                raise ValueError("heatmap layer video_id and segment_id must match manifest")
        return self


class CatalogRecord(BaseModel):
    video_id: str
    date: str
    source_description: str
    video_url: str
    thumbnail_url: str | None = None
    current_stem: str | None = None
    target_stem: str | None = None
    manifest_confidence: str | None = None
    manifest_notes: str | None = None


class CatalogSnapshot(BaseModel):
    readme_source: str
    manifest_source: str | None = None
    fetched_at: str
    row_count: int = Field(ge=0)
    readme_sha256: str
    manifest_sha256: str | None = None


class MediaInventoryRecord(BaseModel):
    video_id: str
    source_url: str
    local_path: Path
    sha256: str
    bytes: int = Field(ge=0)
    duration_sec: float = Field(ge=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class SegmentDiagnostics(BaseModel):
    sampled_frames: int = Field(ge=0)
    brightness_mean: float | None = None
    blur_mean: float | None = None
    motion_mean: float | None = None


class SegmentAnnotation(BaseModel):
    video_id: str
    segment_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    status: Literal["proposed", "accepted", "rejected"]
    include_terminal_event: bool = False
    excluded_ranges_sec: list[list[float]] = Field(default_factory=list)
    annotation_confidence: Literal["low", "medium", "high"] = "medium"
    annotation_notes: str = ""
    diagnostics: SegmentDiagnostics | None = None

    @field_validator("end_sec")
    @classmethod
    def end_after_start(cls, value: float, info: Any) -> float:
        start = info.data.get("start_sec")
        if start is not None and value <= start:
            raise ValueError("end_sec must be greater than start_sec")
        return value


class BundleMetadata(BaseModel):
    schema_version: str
    video_id: str
    segment_id: str
    source_tool: Literal[
        "huggingface-space",
        "colab",
        "runpod",
        "local-vggt",
        "mock",
    ]
    source_url_or_repo: str | None = None
    source_commit_or_version: str | None = None
    export_notes: str | None = None
    generated_at: str
    frame_indices: list[int]
    frame_timestamps_sec: list[float]
    warnings: list[str] = Field(default_factory=list)
    coordinate_frame: str = "VGGT relative coordinate frame"

    @field_validator("frame_indices")
    @classmethod
    def frame_indices_are_ordered(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("frame_indices must not be empty")
        if any(index < 0 for index in value):
            raise ValueError("frame_indices must be non-negative")
        if any(next_index <= index for index, next_index in zip(value, value[1:])):
            raise ValueError("frame_indices must be strictly increasing")
        return value

    @field_validator("frame_timestamps_sec")
    @classmethod
    def timestamps_align_to_indices(
        cls, value: list[float], info: Any
    ) -> list[float]:
        indices = info.data.get("frame_indices", [])
        if len(value) != len(indices):
            raise ValueError("frame_timestamps_sec must align to frame_indices")
        if any(timestamp < 0 for timestamp in value):
            raise ValueError("frame_timestamps_sec must be non-negative")
        if any(next_timestamp < timestamp for timestamp, next_timestamp in zip(value, value[1:])):
            raise ValueError("frame_timestamps_sec must be non-decreasing")
        return value


class ReliabilitySummary(BaseModel):
    score: float = Field(ge=0, le=1)
    label: Literal["good", "mixed", "poor", "failed"]
    failure_flags: list[str] = Field(default_factory=list)
    safe_to_use_for_descriptors: bool


class PathDescriptors(BaseModel):
    sampled_frame_count: int = Field(ge=0)
    valid_pose_count: int = Field(ge=0)
    normalized_path_length: float | None = None
    displacement_ratio: float | None = None
    mean_turn_angle_rad: float | None = None
    max_turn_angle_rad: float | None = None
    pose_jump_count: int = Field(ge=0)


class ReconstructionSummary(BaseModel):
    video_id: str
    segment_id: str
    reliability: ReliabilitySummary
    descriptors: PathDescriptors
    warnings: list[str] = Field(default_factory=list)


class VggtValidationReport(BaseModel):
    valid: bool
    bundle_path: Path
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class PoseSmoothingMetadata(BaseModel):
    video_id: str
    segment_id: str
    label: str = "experimental VGGT-frame relative pose smoother"
    not_physical_dynamics: bool = True
    coordinate_frame: str = "VGGT relative coordinate frame"
    warnings: list[str] = Field(default_factory=list)
