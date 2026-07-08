import json
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.method_contract import (
    CORE_RETURN_REPORT_FILES,
    REQUIRED_METHOD_IDS,
    expanded_method_matrix,
    expanded_return_artifact_files,
    expanded_return_contract,
    validate_method_matrix,
    validate_return_layout,
    write_method_contract_files,
)


runner = CliRunner()


def test_method_contract_files_validate(tmp_path: Path):
    output = tmp_path / "contract"
    artifacts = write_method_contract_files(output)

    assert (output / "METHOD_CONTRACT.md").exists()
    assert (output / "method_matrix.json").exists()
    assert (output / "return_contract.json").exists()
    assert artifacts["method_matrix"].endswith("method_matrix.json")

    report = validate_method_matrix(output / "method_matrix.json")
    assert report["valid"], report["issues"]
    assert set(REQUIRED_METHOD_IDS).issubset(set(report["method_ids"]))


def test_return_layout_validator_accepts_compact_return(tmp_path: Path):
    root = tmp_path / "return"
    root.mkdir()
    write_method_contract_files(root)
    for name in ["manifest.json", "cloud_run.log", "environment.json", "expanded_runtime_check.json", "method_stage_report.json"]:
        (root / name).write_text("{}", encoding="utf-8")
    selected = root / "selected" / "video-001" / "segment-001"
    selected.mkdir(parents=True)
    (selected / "metadata.json").write_text("{}", encoding="utf-8")
    (root / "tier_report.json").write_text("[]", encoding="utf-8")

    report = validate_return_layout(root)
    assert report["valid"], report["issues"]


def test_return_layout_validator_rejects_missing_contract(tmp_path: Path):
    root = tmp_path / "return"
    root.mkdir()
    for name in ["manifest.json", "cloud_run.log", "environment.json", "expanded_runtime_check.json", "return_contract.json", "method_stage_report.json"]:
        (root / name).write_text("{}", encoding="utf-8")
    selected = root / "selected" / "video-001" / "segment-001"
    selected.mkdir(parents=True)
    (selected / "metadata.json").write_text("{}", encoding="utf-8")
    (root / "tier_report.json").write_text("[]", encoding="utf-8")

    report = validate_return_layout(root)
    assert not report["valid"]
    assert any("method_matrix.json" in issue for issue in report["issues"])


def test_return_contract_covers_all_method_artifacts():
    contract = expanded_return_contract()
    optional = set(contract["optional_artifacts"])
    assert "method_stage_report.json" in contract["required_files"]
    assert "method_stage_report.json" not in optional
    assert "odm_artifacts.zip" in optional
    assert "openmvs_artifacts.zip" in optional
    assert "openmvs_dense_mesh_manifest.json" in optional
    assert "nerfstudio_gsplat_showcase.zip" in optional

    one_of = {item for group in contract["required_one_of"] for item in group}
    allowed = set(contract["required_files"]) | optional | one_of | CORE_RETURN_REPORT_FILES
    for method in expanded_method_matrix():
        missing = sorted(set(method["return_contract"]) - allowed)
        assert not missing, f"{method['method_id']} has uncovered artifacts: {missing}"

    assert set(expanded_return_artifact_files()).issubset(optional)


def test_methods_cli_writes_and_validates_contract(tmp_path: Path):
    output = tmp_path / "contract"
    written = runner.invoke(app, ["methods", "write-contract", "--output", str(output)])
    assert written.exit_code == 0, written.output
    assert "method matrix" in written.output

    validated = runner.invoke(
        app,
        ["methods", "validate-matrix", "--matrix", str(output / "method_matrix.json")],
    )
    assert validated.exit_code == 0, validated.output
    payload = json.loads(validated.output)
    assert payload["valid"] is True


def test_methods_cli_validates_return_zip(tmp_path: Path):
    root = tmp_path / "return"
    root.mkdir()
    write_method_contract_files(root)
    for name in ["manifest.json", "cloud_run.log", "environment.json", "expanded_runtime_check.json", "method_stage_report.json"]:
        (root / name).write_text("{}", encoding="utf-8")
    selected = root / "selected" / "video-001" / "segment-001"
    selected.mkdir(parents=True)
    (selected / "metadata.json").write_text("{}", encoding="utf-8")
    (root / "tier_report.json").write_text("[]", encoding="utf-8")
    zip_path = tmp_path / "return.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root))

    result = runner.invoke(app, ["methods", "validate-return", "--source", str(zip_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["valid"] is True



def test_return_layout_validator_rejects_unsafe_zip_member(tmp_path: Path):
    zip_path = tmp_path / "unsafe_return.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../outside.txt", "bad")

    report = validate_return_layout(zip_path)
    assert not report["valid"]
    assert any("unsafe zip member path" in issue for issue in report["issues"])
