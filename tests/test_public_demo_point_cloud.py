from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import typer

import fpv_vggt_lab.public_demo as public_demo
from fpv_vggt_lab.public_demo import (
    PUBLIC_SCHEMA_VERSION,
    PublicDemoConfig,
    PublicDemoError,
    audit_public_demo,
    build_public_demo,
    main,
)
from tests.test_public_demo import _synthetic_sources


AUTHORIZATION = "Dataset maintainer approval reported by the project owner on 2026-07-11"
ATTRIBUTION = "Itamar Weiss / FPV Drone Strikes Open Dataset"


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _authorized_config(
    scene: Path,
    archive: Path,
    output: Path,
    *,
    slug: str = "relative-geometry-study",
) -> PublicDemoConfig:
    return PublicDemoConfig(
        scene_root=scene,
        archive_root=archive,
        output_root=output,
        slug=slug,
        public_title="Relative Geometry Study",
        path_sample_count=16,
        max_density_cells=80,
        publish_real_point_cloud=True,
        point_cloud_authorization=AUTHORIZATION,
        point_cloud_attribution=ATTRIBUTION,
        generated_at="2026-07-11T12:00:00Z",
    )


def test_authorized_build_copies_exact_colored_point_sample_and_contract(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    source_positions = scene / "viewer" / "points_preview.bin"
    source_colors = scene / "viewer" / "points_preview_colors.bin"
    camera_path = json.loads(
        (scene / "viewer" / "camera_path.json").read_text(encoding="utf-8")
    )
    raw_path = np.asarray(camera_path["layers"]["raw"], dtype=np.float64)
    low = np.quantile(raw_path, 0.01, axis=0)
    high = np.quantile(raw_path, 0.99, axis=0)
    expected_center = (low + high) * 0.5
    expected_scale = max(float(np.max(high - low)) * 0.5, 1e-9)

    result = build_public_demo(_authorized_config(scene, archive, output))

    scene_dir = output / "scenes" / "relative-geometry-study"
    published_positions = scene_dir / "geometry" / "vggt_omega_points.f32.bin"
    published_colors = scene_dir / "geometry" / "vggt_omega_colors.rgb8.bin"
    payload = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    point_cloud = payload["point_cloud"]

    assert PUBLIC_SCHEMA_VERSION == "1.1.0"
    assert payload["schema_version"] == "1.1.0"
    assert published_positions.read_bytes() == source_positions.read_bytes()
    assert published_colors.read_bytes() == source_colors.read_bytes()
    assert point_cloud["backend"] == "VGGT Omega"
    assert point_cloud["scale_status"] == "relative_only"
    assert point_cloud["point_count"] == 9000
    assert point_cloud["source_point_count"] == 9000
    assert point_cloud["positions"] == {
        "path": "geometry/vggt_omega_points.f32.bin",
        "dtype": "float32_le",
        "components": 3,
        "bytes": source_positions.stat().st_size,
        "sha256": hashlib.sha256(source_positions.read_bytes()).hexdigest(),
    }
    assert point_cloud["colors"] == {
        "path": "geometry/vggt_omega_colors.rgb8.bin",
        "dtype": "uint8",
        "components": 3,
        "bytes": source_colors.stat().st_size,
        "sha256": hashlib.sha256(source_colors.read_bytes()).hexdigest(),
    }
    assert point_cloud["display_transform"]["center"] == pytest.approx(expected_center)
    assert point_cloud["display_transform"]["scale"] == pytest.approx(expected_scale)
    assert point_cloud["authorization_provenance"] == AUTHORIZATION
    assert point_cloud["attribution"] == ATTRIBUTION
    assert manifest["point_cloud"] == point_cloud
    assert payload["publication_boundary"]["real_point_sample_published"] is True
    assert payload["publication_boundary"]["full_backend_arrays_published"] is False
    for key in (
        "original_video_published",
        "source_frames_published",
        "recognizable_reprojection_published",
        "raw_npz_or_ply_published",
        "absolute_source_paths_published",
    ):
        assert payload["publication_boundary"][key] is False

    manifest_files = {record["path"]: record for record in manifest["files"]}
    for published, source in (
        (published_positions, source_positions),
        (published_colors, source_colors),
    ):
        relative = published.relative_to(output).as_posix()
        assert manifest_files[relative]["bytes"] == source.stat().st_size
        assert manifest_files[relative]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_default_rebuild_removes_known_authorized_point_binaries(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    build_public_demo(_authorized_config(scene, archive, output))
    geometry_dir = output / "scenes" / "relative-geometry-study" / "geometry"
    retained_text = geometry_dir / "review-note.txt"
    retained_text.write_text("retain non-binary output", encoding="utf-8")

    result = build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            path_sample_count=16,
            max_density_cells=80,
            generated_at="2026-07-11T12:05:00Z",
        )
    )

    payload = json.loads(
        (output / "scenes" / "relative-geometry-study" / "scene.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["publication_boundary"]["real_point_sample_published"] is False
    assert "point_cloud" not in payload
    assert "point_cloud" not in manifest
    assert not list(output.rglob("*.bin"))
    assert retained_text.read_text(encoding="utf-8") == "retain non-binary output"


def test_build_rejects_preexisting_forbidden_binary_anywhere(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    rogue = output / "unrelated" / "nested" / "rogue.bin"
    rogue.parent.mkdir(parents=True)
    rogue.write_bytes(b"not an authorized public asset")

    with pytest.raises(PublicDemoError, match="forbidden public file type"):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=output,
                slug="relative-geometry-study",
                public_title="Relative Geometry Study",
                generated_at="2026-07-11T12:10:00Z",
            )
        )

    assert rogue.read_bytes() == b"not an authorized public asset"


def test_cross_slug_default_build_rejects_stale_authorized_binaries(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    build_public_demo(
        _authorized_config(scene, archive, output, slug="authorized-alpha")
    )
    stale_position = (
        output
        / "scenes"
        / "authorized-alpha"
        / "geometry"
        / "vggt_omega_points.f32.bin"
    )

    with pytest.raises(PublicDemoError, match="forbidden public file type"):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=output,
                slug="default-beta",
                public_title="Default Beta",
                generated_at="2026-07-11T12:15:00Z",
            )
        )

    assert stale_position.is_file()


def test_build_rejects_preexisting_stale_text_leak_without_mutation(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    stale = output / "legacy" / "stale.html"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        '<script src="https://cdn.example.invalid/runtime.js"></script>'
        '<a href="D:\\private\\source.mp4">leak</a>',
        encoding="utf-8",
    )
    before = _tree_bytes(output)

    with pytest.raises(PublicDemoError, match="public output audit failed"):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=output,
                slug="relative-geometry-study",
                public_title="Relative Geometry Study",
                generated_at="2026-07-11T12:20:00Z",
            )
        )

    assert _tree_bytes(output) == before


def test_build_excludes_git_internals_from_effective_public_audit(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    git_internal = output / ".git" / "audit-fixture.html"
    git_internal.parent.mkdir(parents=True)
    git_internal.write_text('<a href="D:\\private\\ignored">ignored</a>', encoding="utf-8")

    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            generated_at="2026-07-11T12:25:00Z",
        )
    )

    assert git_internal.read_text(encoding="utf-8") == (
        '<a href="D:\\private\\ignored">ignored</a>'
    )


def test_late_template_audit_failure_leaves_prior_output_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Original Study",
            generated_at="2026-07-11T12:30:00Z",
        )
    )
    before = _tree_bytes(output)
    original_render = public_demo._render_template

    def render_with_late_leak(path: Path, replacements: dict[str, str]) -> str:
        rendered = original_render(path, replacements)
        if path.name == "gallery.html":
            return f'{rendered}\n<a href="D:\\private\\late-leak">leak</a>'
        return rendered

    monkeypatch.setattr(public_demo, "_render_template", render_with_late_leak)

    with pytest.raises(PublicDemoError, match="public output audit failed"):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=output,
                slug="relative-geometry-study",
                public_title="Changed Study",
                generated_at="2026-07-11T12:35:00Z",
            )
        )

    assert _tree_bytes(output) == before


def test_transactional_publish_failure_restores_prior_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Original Study",
            generated_at="2026-07-11T12:40:00Z",
        )
    )
    before = _tree_bytes(output)
    real_replace = os.replace
    publish_calls = 0
    injected = False

    def replace_with_one_publish_failure(source: str | Path, destination: str | Path) -> None:
        nonlocal publish_calls, injected
        destination_path = Path(destination).resolve()
        if not injected and destination_path.is_relative_to(output.resolve()):
            publish_calls += 1
            if publish_calls == 3:
                injected = True
                raise OSError("injected transactional publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace_with_one_publish_failure)

    with pytest.raises(PublicDemoError, match="transactional public publish failed"):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=output,
                slug="relative-geometry-study",
                public_title="Changed Study",
                generated_at="2026-07-11T12:45:00Z",
            )
        )

    assert publish_calls == 3
    assert _tree_bytes(output) == before


@pytest.mark.parametrize(
    ("authorization", "attribution", "message"),
    [
        (None, ATTRIBUTION, "authorization"),
        ("   ", ATTRIBUTION, "authorization"),
        (AUTHORIZATION, None, "attribution"),
        (AUTHORIZATION, "   ", "attribution"),
    ],
)
def test_real_point_publication_requires_authorization_and_attribution(
    tmp_path: Path,
    authorization: str | None,
    attribution: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PublicDemoConfig(
            scene_root=tmp_path / "scene",
            archive_root=tmp_path / "archive",
            output_root=tmp_path / "docs",
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            publish_real_point_cloud=True,
            point_cloud_authorization=authorization,
            point_cloud_attribution=attribution,
        )


def test_real_point_publication_rejects_mismatched_color_count(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    colors = scene / "viewer" / "points_preview_colors.bin"
    colors.write_bytes(colors.read_bytes()[:-1])

    with pytest.raises(PublicDemoError, match="RGB8"):
        build_public_demo(_authorized_config(scene, archive, tmp_path / "docs"))


def test_real_point_publication_rejects_missing_colors(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    (scene / "viewer" / "points_preview_colors.bin").unlink()

    with pytest.raises(PublicDemoError, match="colors"):
        build_public_demo(_authorized_config(scene, archive, tmp_path / "docs"))


def test_real_point_publication_rejects_nonfinite_xyz(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    positions = scene / "viewer" / "points_preview.bin"
    values = np.fromfile(positions, dtype="<f4")
    values[0] = np.nan
    values.tofile(positions)

    with pytest.raises(PublicDemoError, match="finite"):
        build_public_demo(_authorized_config(scene, archive, tmp_path / "docs"))


def test_real_point_publication_rejects_xyz_trailing_bytes(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    positions = scene / "viewer" / "points_preview.bin"
    positions.write_bytes(positions.read_bytes() + b"\x00")

    with pytest.raises(PublicDemoError, match="byte length.*divisible by 12"):
        build_public_demo(_authorized_config(scene, archive, tmp_path / "docs"))


def test_public_audit_allows_only_declared_point_binary_paths(tmp_path: Path) -> None:
    output = tmp_path / "docs"
    allowed_relative = "scenes/study/geometry/vggt_omega_points.f32.bin"
    allowed_path = output / allowed_relative
    allowed_path.parent.mkdir(parents=True)
    allowed_path.write_bytes(b"declared point bytes")

    allowed = audit_public_demo(output, authorized_binary_paths=(allowed_relative,))

    assert allowed["status"] == "passed"

    arbitrary = output / "scenes" / "study" / "geometry" / "arbitrary.bin"
    arbitrary.write_bytes(b"undeclared bytes")
    undeclared = audit_public_demo(output, authorized_binary_paths=(allowed_relative,))

    assert undeclared["status"] == "failed"
    assert {
        (finding["path"], finding["reason"]) for finding in undeclared["findings"]
    } == {("scenes/study/geometry/arbitrary.bin", "forbidden public file type")}


def test_public_demo_cli_exposes_explicit_point_cloud_opt_in() -> None:
    app = typer.Typer()
    app.command()(main)
    command = typer.main.get_command(app)

    options = {
        option
        for parameter in command.params
        for option in getattr(parameter, "opts", ())
    }

    assert "--publish-real-point-cloud" in options
    assert "--point-cloud-authorization" in options
    assert "--point-cloud-attribution" in options


def test_public_demo_cli_defaults_to_isolated_build_root() -> None:
    app = typer.Typer()
    app.command()(main)
    command = typer.main.get_command(app)
    output_parameter = next(
        parameter for parameter in command.params if "--output-root" in parameter.opts
    )

    assert output_parameter.default == Path("build/public_demo")
