# RunPod Expanded Environment

This is the environment recipe for the expanded full-dataset run:

- VGGT feed-forward inference
- VGGT official COLMAP export / bundle adjustment
- classical COLMAP baseline
- relative trajectory export for EVO-style review
- single-run post-QC stages for ODM, Nerfstudio/gsplat, relative-depth overlays, MASt3R/DUSt3R, and VGGeTR when runtimes are available

For the cheaper high-end GPU path, use `docs/runpod_4090_full_stack.md`.

## Current Upload Package

Use the validated full package:

```text
outputs/h100/full_161_run/runpod_job_expanded_methods_store.zip
```

Do not use `runpod_job_expanded_methods.zip`; it is stale and does not contain
`scripts/05_expanded_runtime_check.py`.

## Image Recipe

Template:

```text
containers/runpod-vggt-colmap.Dockerfile
```

Build it once, push it to a registry, and run RunPod by immutable image digest.
For a paid production run, set `VGGT_REF` to a VGGT commit SHA rather than
`main`.

The image sets `VGGT_REPO_DIR=/workspace/vggt`; keep that path unless you also
set `VGGT_REPO_DIR` explicitly in the pod environment.

Example build:

```bash
docker build \
  -f containers/runpod-vggt-colmap.Dockerfile \
  --build-arg VGGT_REF=<vggt_commit_sha> \
  -t <registry>/fpv-vggt-colmap:<tag> \
  .
```

Optional showcase build:

```bash
docker build \
  -f containers/runpod-vggt-colmap.Dockerfile \
  --build-arg VGGT_REF=<vggt_commit_sha> \
  --build-arg INSTALL_SHOWCASE_TOOLS=true \
  -t <registry>/fpv-vggt-colmap-showcase:<tag> \
  .
```

Research-method build:

```bash
docker build \
  -f containers/runpod-vggt-colmap.Dockerfile \
  --build-arg VGGT_REF=<vggt_commit_sha> \
  --build-arg DUST3R_REF=<dust3r_commit_sha> \
  --build-arg MAST3R_REF=<mast3r_commit_sha> \
  --build-arg INSTALL_RESEARCH_METHODS=true \
  -t <registry>/fpv-vggt-colmap-research:<tag> \
  .
```

Use `COMPILE_RESEARCH_CUDA_KERNELS=true` only after the base research image builds cleanly. DUSt3R and MASt3R are CC BY-NC-SA 4.0 projects; keep this as non-commercial research/portfolio work and pin commits for paid runs.

Use OpenMVS as the preferred dense reconstruction baseline when you intentionally build a larger photogrammetry image. Keep ODM as a separate optional post-QC runtime unless you need to compare against it specifically. ODM outputs must stay local/relative and must not
be treated as geolocation or map projection.

## Pod Preflight

After extracting the package on RunPod, run the runtime and dataset checks before `run_all.sh`:

```bash
cd /workspace/fpv-expanded
python scripts/05_expanded_runtime_check.py
python scripts/06_hf_dataset_preflight.py
python scripts/07_hf_dataset_frame_packs.py
cat expanded_runtime_check.json
cat hf_dataset_report.json
cat hf_frame_pack_report.json
```

For the private Hugging Face dataset path, set `HF_DATASET_ID=Grimster/FPV_Hezbo`
and `HF_TOKEN` on the pod. Set `REQUIRE_HF_DATASET=1` when the run must stop if
that dataset cannot be accessed. Set `HF_DATASET_DOWNLOAD=1` only when you want
the pod to mirror the dataset snapshot under `/workspace/hf_datasets`; the
current full package still runs from packaged frame packs unless a future package
maps the snapshot into frame manifests. Set `HF_DATASET_BUILD_FRAME_PACKS=1` to let `scripts/07_hf_dataset_frame_packs.py` build replacement frame packs from supported image folders or video files.

Required readiness for the main run:

- `vggt_feedforward_ready: true`
- `vggt_colmap_ba_ready: true`
- `classical_colmap_ready: true` if you want the classical baseline

Optional readiness:

- `trajectory_evo_ready`
- `openmvs_ready`
- `odm_ready`
- `nerfstudio_gsplat_ready`
- `research_methods_ready`
- `dust3r_builtin_runner_ready`
- `mast3r_builtin_runner_ready`
- `vggetr_template_ready`

`expanded_runtime_check.json` also includes `research_method_runners`, which records the exact DUSt3R/MASt3R import-probe result and command-template flags before any model checkpoint download or inference begins.

The preflight also writes:

- `readiness_status`: `ready_full_stack`, `ready_4090_pilot`, or `blocked_primary_runtime`
- `missing_primary_checks`: primary blockers to fix before the full run
- `missing_optional_checks`: optional methods that will likely be skipped
- `next_actions`: plain-language fixes for the current pod image

If `readiness_status` is `blocked_primary_runtime`, stop before `bash run_all.sh` and fix the VGGT runtime/image first. `ready_4090_pilot` is enough for a bounded VGGT feed-forward pilot, while `ready_full_stack` is required before claiming VGGT+COLMAP/classical COLMAP all ran. Optional missing methods are allowed to continue and should be recorded as `skipped_missing_dependency` in `method_stage_report.json`.

## Official Method Notes

VGGT documents COLMAP export through `demo_colmap.py`:

```bash
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba
```

The scene must contain images under:

```text
SCENE_DIR/images/
```

VGGT writes COLMAP sparse reconstruction files under:

```text
SCENE_DIR/sparse/
```

Classical COLMAP is still a separate baseline. The package uses the command
sequence:

```bash
colmap feature_extractor
colmap sequential_matcher
colmap mapper
```


## Single-Run Research Methods

The expanded run now treats MASt3R/DUSt3R/VGGeTR as one cloud-job stage, not as a later manual adapter pass.

- DUSt3R: if the `dust3r` Python module is importable, the package writes and runs a built-in Python runner using `dust3r.inference`, `make_pairs`, and global alignment.
- MASt3R: if the `mast3r` Python module is importable, the package writes and runs a built-in Python runner using the MASt3R model plus DUSt3R pair/global-alignment utilities.
- VGGeTR/VG2GT: still requires `VGGETR_COMMAND_TEMPLATE`, `VG2GT_COMMAND_TEMPLATE`, or `RESEARCH_METHOD_COMMAND_TEMPLATE` until the exact public repo/checkpoint/API is confirmed.

Useful environment knobs:

```bash
export OPTIONAL_SHOWCASE_MAX_CLIPS=10        # raise this to run more post-QC windows
export RESEARCH_METHOD_MAX_IMAGES=64        # deterministic even sample inside each research-method project
export RESEARCH_METHOD_MAX_POINTS=200000    # cap returned point samples per method
export DUST3R_MODEL_NAME=naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
export MAST3R_MODEL_NAME=naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
# export VG2GT_COMMAND_TEMPLATE="python /workspace/vg2gt/run.py --images {raw_images} --out {output_dir}"
# export SKIP_RESEARCH_METHODS=1            # only if you want to disable this stage
```

All outputs remain relative diagnostics only: no meters, no coordinates, no route/approach interpretation.

## Return Inspection

After downloading the return package:

```powershell
fpv h100 inspect-return `
  --source D:\Drone_Analysis_Dynamics\outputs\h100_returns\<new_return>.zip
```

Machine-readable form:

```powershell
fpv h100 inspect-return `
  --source D:\Drone_Analysis_Dynamics\outputs\h100_returns\<new_return>.zip `
  --json
```

Optional-method quick report after download:

```powershell
fpv h100 optional-report `
  --source D:\Drone_Analysis_Dynamics\outputs\h100_returns\<new_return>.zip `
  --output-dir outputs/reviews/h100_optional_methods
```
The JSON includes `expanded_runtime_check`, `method_stage_report`,
`quality_report`, expanded artifact inventory, validation `issues`, and `next_actions`.

During import, returned COLMAP sparse archives, trajectory exports, EVO reports,
and optional-method archives are copied to `expanded_artifacts/` inside the local
review output.

## Safety Boundary

All outputs remain offline, local-only, and relative-frame diagnostics:

- no geolocation
- no map projection for real videos
- no meters or true speed/standoff/dive-angle claims
- no route, launch, target-coordinate, guidance, or next-maneuver inference

## Verified Package Hashes

- Full upload ZIP: `outputs/h100/full_161_run/runpod_job_expanded_methods_store.zip`
- Full upload size: `3,316,091,860 bytes`
- Full upload SHA256: `5f6909e4361128751baf8564444fcf72a953581cac34fe9e7cc2bcb09d0b1a19`
- Patch ZIP: `outputs/h100/full_161_run/runpod_job_patch_expanded_methods.zip`
- Patch size: `74,209 bytes`
- Patch SHA256: `83b28370f72988cf71ba0a7adb5f450cf725dd0c92b133013222c377ea74d377`
- Verified on: `2026-07-05`
