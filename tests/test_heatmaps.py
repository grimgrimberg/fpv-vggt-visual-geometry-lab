from pathlib import Path

import cv2
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.frames import sample_video_frames
from fpv_vggt_lab.heatmaps import generate_heatmaps_from_manifest, read_heatmap_manifest
from fpv_vggt_lab.schemas import HeatmapLayerRecord, HeatmapManifest, model_to_dict
from fpv_vggt_lab.synthetic import create_synthetic_video


runner = CliRunner()


def test_heatmap_manifest_schema_requires_image_space_layers(tmp_path: Path):
    frame_path = tmp_path / "frame.jpg"
    heatmap_path = tmp_path / "heatmap.png"
    frame_path.write_bytes(b"frame")
    heatmap_path.write_bytes(b"heatmap")

    record = HeatmapLayerRecord(
        video_id="video-a",
        segment_id="segment-001",
        frame_index=7,
        timestamp_sec=1.4,
        layer_name="optical_flow",
        layer_kind="image_space",
        source_frame_path=frame_path,
        heatmap_path=heatmap_path,
        width=160,
        height=120,
        value_min=0.0,
        value_max=1.0,
        normalization="per-layer-frame min-max to uint8",
        warnings=[],
        local_only=True,
    )
    manifest = HeatmapManifest(
        video_id="video-a",
        segment_id="segment-001",
        source_frame_manifest=tmp_path / "frames.json",
        layers=[record],
        warnings=["image-space diagnostics only", "no geolocation", "no meters"],
    )

    payload = model_to_dict(manifest)
    assert payload["layers"][0]["layer_kind"] == "image_space"
    assert payload["layers"][0]["local_only"] is True
    assert "no geolocation" in payload["warnings"]


def _heatmap_layer_record(tmp_path: Path, **overrides: object) -> HeatmapLayerRecord:
    values = {
        "video_id": "video-a",
        "segment_id": "segment-001",
        "frame_index": 7,
        "timestamp_sec": 1.4,
        "layer_name": "optical_flow",
        "layer_kind": "image_space",
        "source_frame_path": tmp_path / "frame.jpg",
        "heatmap_path": tmp_path / "heatmap.png",
        "width": 160,
        "height": 120,
        "value_min": 0.0,
        "value_max": 1.0,
        "normalization": "per-layer-frame min-max to uint8",
        "warnings": [],
        "local_only": True,
    }
    values.update(overrides)
    return HeatmapLayerRecord(**values)


def test_heatmap_layer_rejects_non_local_output(tmp_path: Path):
    with pytest.raises(ValidationError, match="local-only"):
        _heatmap_layer_record(tmp_path, local_only=False)


def test_heatmap_manifest_rejects_empty_layers(tmp_path: Path):
    with pytest.raises(ValidationError, match="at least one layer"):
        HeatmapManifest(
            video_id="video-a",
            segment_id="segment-001",
            source_frame_manifest=tmp_path / "frames.json",
            layers=[],
            warnings=[],
        )


def test_heatmap_layer_rejects_invalid_value_range(tmp_path: Path):
    with pytest.raises(ValidationError, match="value_min must be less than or equal"):
        _heatmap_layer_record(tmp_path, value_min=2.0, value_max=1.0)


def test_heatmap_manifest_rejects_layer_from_different_video_or_segment(
    tmp_path: Path,
):
    record = _heatmap_layer_record(
        tmp_path,
        video_id="video-b",
        segment_id="segment-999",
    )

    with pytest.raises(ValidationError, match="must match manifest"):
        HeatmapManifest(
            video_id="video-a",
            segment_id="segment-001",
            source_frame_manifest=tmp_path / "frames.json",
            layers=[record],
            warnings=[],
        )


def test_heatmap_layer_rejects_extra_geolocation_field(tmp_path: Path):
    with pytest.raises(ValidationError, match="Extra inputs"):
        _heatmap_layer_record(tmp_path, latitude=32.0)


def test_heatmap_layer_rejects_invalid_layer_kind(tmp_path: Path):
    with pytest.raises(ValidationError, match="Input should be 'image_space'"):
        _heatmap_layer_record(tmp_path, layer_kind="map_projection")


@pytest.mark.parametrize("field_name", ["value_min", "value_max"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_heatmap_layer_rejects_nan_and_infinity_values(
    tmp_path: Path,
    field_name: str,
    bad_value: float,
):
    with pytest.raises(ValidationError, match="finite"):
        _heatmap_layer_record(tmp_path, **{field_name: bad_value})


def test_generate_heatmaps_from_synthetic_frame_manifest(tmp_path: Path):
    video_path = tmp_path / "synthetic.mp4"
    frames_dir = tmp_path / "frames"
    heatmap_dir = tmp_path / "heatmaps"
    create_synthetic_video(video_path, frames=18, width=160, height=120)
    frame_manifest = sample_video_frames(
        video=video_path,
        output=frames_dir,
        count=6,
        video_id="heatmap-video",
        segment_id="segment-001",
    )

    manifest_path = generate_heatmaps_from_manifest(
        frame_manifest_path=frames_dir / "frames.json",
        output_dir=heatmap_dir,
    )

    manifest = read_heatmap_manifest(manifest_path)
    layer_names = {layer.layer_name for layer in manifest.layers}
    assert layer_names == {
        "frame_difference",
        "optical_flow",
        "blur",
        "visibility_change",
    }
    assert len(manifest.layers) == len(frame_manifest.frames) * 4
    assert "no geolocation" in manifest.warnings

    first_flow = next(
        layer for layer in manifest.layers if layer.layer_name == "optical_flow"
    )
    image = cv2.imread(str(first_flow.heatmap_path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape[:2] == (first_flow.height, first_flow.width)
    assert image.shape[2] == 4
    assert image[..., 3].max() > 0


def test_heatmap_generate_cli_writes_manifest(tmp_path: Path):
    video_path = tmp_path / "synthetic-cli.mp4"
    frames_dir = tmp_path / "frames-cli"
    heatmap_dir = tmp_path / "heatmaps-cli"
    create_synthetic_video(video_path, frames=16, width=160, height=120)
    sample_video_frames(
        video=video_path,
        output=frames_dir,
        count=5,
        video_id="heatmap-cli-video",
        segment_id="segment-001",
    )

    result = runner.invoke(
        app,
        [
            "heatmap",
            "generate",
            "--frame-manifest",
            str(frames_dir / "frames.json"),
            "--output",
            str(heatmap_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "wrote heatmap manifest" in result.output
    manifest = read_heatmap_manifest(heatmap_dir / "heatmaps.json")
    assert len(manifest.layers) == 20
