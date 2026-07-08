# RunPod 4090 Full-Stack Goal

Status: `ready_to_run_4090_pilot`

Created: 2026-07-06

## Objective

Run the full offline FPV visual-geometry pipeline from a private Hugging Face
dataset source, using a cheaper high-end RunPod GPU profile first, and return a
complete local review package.

Naming note: `h100_return.zip`, `fpv h100`, and `outputs/h100/...` are legacy internal contract names kept for compatibility. They do not mean the current run used an H100; this launch targets the 4090/HF pod profile unless you explicitly choose a different GPU.

Primary target:

- RTX 4090 24 GB

Fallback targets:

- RTX 6000 Ada / L40S / A6000 48 GB when 4090 memory is not enough
- H100 only when the cheaper profiles fail after a documented pilot

Dataset source:

- `Grimster/FPV_Hezbo`
- Hugging Face status checked on 2026-07-06: private dataset
- Access requires an HF token on the cloud pod

## Current Standing

What is ready locally:

- Full expanded RunPod package exists:
  `outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip`
- Package SHA256:
  `1d36a3968b6dc64664265120e85489d120ede5b1e33032d7088047bbe97c232d`
- Package inspection passed with no issues.
- Machine-readable 4090 pod profile exists:
  `outputs/h100/full_161_run/launch/RUNPOD_4090_PILOT_PROFILE.json`.
- Source-owned 4090/HF launch status and audit commands are available:
  `fpv runpod status-4090-launch` and `fpv runpod audit-4090-launch`.
- Source-owned 4090/HF launch audit passed with full package SHA verification:
  `fpv runpod audit-4090-launch`.
- Operator prelaunch helper passed in fast mode:
  `outputs/h100/full_161_run/launch/PRELAUNCH_AUDIT_4090_HF.ps1 -Fast`.
- Focused package/return tests passed.
- Existing local returns were audited in
  `outputs/h100/full_161_run/RETURN_STATUS_AUDIT.md`, but that audit is not
  proof of the current 4090/HF expanded run.

What is not complete:

- No fresh expanded-method return has been validated locally.
- The June 28 return has 161 compact VGGT bundles only.
- The older large `outputs/h100_returns/h100_return.zip` is not a readable ZIP.
- COLMAP BA, classical COLMAP, EVO, ODM, Nerfstudio/gSplat, MASt3R/DUSt3R,
  VGGeTR/VG2GT, and relative-depth overlays are not proven complete until a new
  expanded return is downloaded and validated.

## 4090 Strategy

Runtime readiness is split: `ready_4090_pilot` is enough for a bounded VGGT feed-forward pilot on a 4090, while `ready_full_stack` is required before claiming VGGT+COLMAP/classical COLMAP all ran.



Use the 4090 as the default production-cost target, but do not ask it to behave
like an H100.

Recommended first 4090 profile:

- Run only one GPU process at a time.
- Use sequential clip processing.
- Keep `DUST3R_BATCH_SIZE=1` and `MAST3R_BATCH_SIZE=1`.
- Use `RESEARCH_METHOD_MAX_IMAGES=32` for the first pilot.
- Use `OPTIONAL_SHOWCASE_MAX_CLIPS=3` for the first pilot.
- Keep ODM and Nerfstudio execution disabled unless the primary VGGT/COLMAP
  stages pass.

Recommended full 4090 profile:

- Scout tier: 32 frames.
- Main tier: 64 frames.
- High-detail tier: 96 frames.
- Resize long edge: 768 to 1024 depending on VRAM preflight.
- Increase high-detail frames only after the pilot proves stable.

Use a 48 GB GPU when:

- 4090 runs out of memory on VGGT high-detail windows.
- VGGT COLMAP BA fails due memory rather than geometry.
- Nerfstudio/gSplat is being executed for many top windows.

## Required Cloud Preflight

Before spending a long run, the pod must prove:

- CUDA PyTorch works.
- `vggt` imports.
- VGGT weights are accessible or cached.
- `VGGT_REPO_DIR` points to a checkout containing `demo_colmap.py`.
- `colmap` is on `PATH`.
- `python -m zipfile` can extract the package.
- `scripts/06_hf_dataset_preflight.py` writes `hf_dataset_report.json`.
- `scripts/07_hf_dataset_frame_packs.py` can build replacement frame packs from a downloaded/local HF snapshot when explicitly enabled.
- `huggingface_hub` can access `Grimster/FPV_Hezbo` using `HF_TOKEN` when `HF_DATASET_ID` is set.
- `REQUIRE_HF_DATASET=1` blocks before expensive inference if HF access is missing.
- Disk volume has enough room for frames, raw predictions, COLMAP sparse
  outputs, logs, return archives, and partial failure packages.

The run must stop before expensive inference if primary checks fail.

## One-Run Contract

The cloud run must produce `h100_return.zip` even when stages fail softly.

Required return evidence:

- `manifest.json`
- `cloud_run.log`
- `environment.json`
- `expanded_runtime_check.json`
- `hf_dataset_report.json` when the HF preflight stage runs
- `hf_frame_pack_report.json` when the optional HF frame-pack stage runs
- `method_stage_report.json`
- `quality_report.json`
- selected compact VGGT bundles or a clear failure report
- method artifact audit data after local validation

Expected expanded artifacts when available:

- VGGT raw predictions
- VGGT + official COLMAP sparse outputs
- converted or preserved COLMAP bundles
- classical COLMAP sparse outputs and logs
- trajectory exports and EVO-style reports
- ODM project reports/logs
- Nerfstudio/gSplat project reports/renders
- MASt3R/DUSt3R/VGGeTR/VG2GT method reports
- relative-depth overlay manifests

## Execution Plan

1. Build or select a pinned RunPod image.
2. Run local status/preflight gates: `fpv runpod status-4090-launch`,
   `fpv runpod audit-4090-launch`, and
   `outputs/h100/full_161_run/launch/PRELAUNCH_AUDIT_4090_HF.ps1`.
3. Upload the refreshed package.
4. Run the package preflight on the pod.
5. If the 4090 profile passes, run the default 3-item scout pilot.
6. Validate the pilot return locally.
7. If the pilot passes, run the full dataset on the same 4090 profile.
8. If the pilot fails from memory, move to L40S / RTX 6000 Ada / A6000 48 GB.
9. Download `h100_return.zip`.
10. Run `VALIDATE_RETURN.ps1`.
11. Import locally and render local review artifacts.

## Success Criteria

The goal is complete only when:

- the private HF dataset source is either consumed directly or mirrored into the
  local package with provenance recorded,
- the chosen RunPod GPU profile is recorded,
- a fresh return ZIP is downloaded,
- local validation passes or produces a complete failed-soft package,
- local review artifacts are rendered,
- method-stage statuses explain what ran, skipped, or failed,
- no output makes geolocation, route, meter, speed, standoff, guidance, or
  next-maneuver claims.

## Safety Boundary

This remains offline historical-media reconstruction QA. No geolocation. No meters. No route or guidance claims.

Do not add:

- geolocation
- map projection
- route or approach analysis
- launch or target coordinate inference
- true speed/standoff/dive-angle claims
- guidance, next-maneuver prediction, or tactical recommendations
