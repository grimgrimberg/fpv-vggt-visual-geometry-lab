# RunPod 4090 Full-Stack Runbook

Use this runbook when the goal is to run the expanded FPV visual-geometry stack
on a cheaper high-end RunPod GPU before paying for H100 time.

## Current Status

Status on 2026-07-07: `ready_to_run_4090_pilot`.

Naming note: `h100_return.zip`, `fpv h100`, and `outputs/h100/...` are legacy internal contract names kept for compatibility. They do not mean the current run used an H100; this launch targets the 4090/HF pod profile unless you explicitly choose a different GPU.

Local package ready:

```text
outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip
```

Package SHA256:

```text
1d36a3968b6dc64664265120e85489d120ede5b1e33032d7088047bbe97c232d
```

Package size:

```text
3375195420 bytes
```

Current 4090/HF launch status:

```text
outputs/h100/full_161_run/launch/runpod_launch_4090_hf_manifest.json
outputs/h100/full_161_run/launch/RUNPOD_LAUNCH_4090_HF.md
outputs/h100/full_161_run/launch/RUNPOD_4090_PILOT_PROFILE.json
```

These files say a fresh cloud return is still required. Do not use the legacy
`outputs/h100/full_161_run/RETURN_STATUS_AUDIT.md`,
`outputs/h100/full_161_run/launch/RUNPOD_LAUNCH.md`, or
`outputs/h100/full_161_run/launch/runpod_launch_manifest.json` for the 4090/HF
package; they describe older expanded-methods launch surfaces.

Canonical local status and readiness commands:

```powershell
fpv runpod status-4090-launch
fpv runpod smoke-package --source outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip
fpv runpod audit-4090-launch
```

Use `status-4090-launch` for plain next steps. Use `audit-4090-launch --fast` for quick iteration and the full audit command before paid upload.

## Hugging Face Dataset Source

The authenticated Hugging Face account is `Grimster`.

Dataset selected for this run:

```text
Grimster/FPV_Hezbo
```

Status checked on 2026-07-06:

```text
private
```

Set `HF_TOKEN` on the pod. Do not write it into repo files, logs, returned
artifacts, or shell history snippets.

The generated package now includes `scripts/06_hf_dataset_preflight.py`. It
writes `hf_dataset_report.json` during `run_all.sh` and returns that report in
`h100_return.zip`. The package also includes `scripts/07_hf_dataset_frame_packs.py`, which can replace packaged frame packs from a downloaded/local HF snapshot when `HF_DATASET_BUILD_FRAME_PACKS=1`. With no `HF_DATASET_ID`, the preflight stage records
`skipped_no_dataset_id` and uses the packaged local frame packs. With
`REQUIRE_HF_DATASET=1`, missing or inaccessible HF data blocks before expensive
inference and still leaves a report.

## Recommended RunPod Profile

Start with:

- GPU: RTX 4090 24 GB
- System RAM: 64 GB or higher
- Container disk: 40 GB or higher
- Volume disk: 150 GB or higher for full-dataset artifacts

Use a 48 GB GPU if the 4090 pilot fails from memory:

- L40S
- RTX 6000 Ada
- A6000

Reserve H100 for the case where the 4090/48 GB profile fails after a documented
pilot.
## Container Image

Preferred path: build `containers/runpod-vggt-colmap.Dockerfile` once, push it to a registry, and run the pod by immutable image digest. The default base is `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, with `colmap`, `ffmpeg`, `unzip`, VGGT, `pycolmap`, `trimesh`, and `evo` installed by the Dockerfile. Build with `--build-arg INSTALL_OPENMVS=true` when you want the recommended OpenMVS dense-mesh baseline to execute instead of returning `openmvs_ready=false`.

For a paid full run, pin `VGGT_REF` to a commit SHA. Enable `INSTALL_RESEARCH_METHODS=true` only after the base VGGT+COLMAP pilot passes, because DUSt3R/MASt3R setup is optional and can make image builds slower or more fragile.

## 4090 Environment Defaults

Set these before running the package:

```bash
export HF_DATASET_ID=Grimster/FPV_Hezbo
# export HF_TOKEN=...   # set interactively or through RunPod secrets, never commit it
export REQUIRE_HF_DATASET=1
export HF_DATASET_DOWNLOAD=1
export HF_DATASET_BUILD_FRAME_PACKS=1
export HF_DATASET_MAX_ITEMS=3
export HF_DATASET_FRAME_PACK_TIERS=scout
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPTIONAL_SHOWCASE_MAX_CLIPS=3
export RESEARCH_METHOD_MAX_IMAGES=32
export RESEARCH_METHOD_MAX_POINTS=100000
export DUST3R_BATCH_SIZE=1
export MAST3R_BATCH_SIZE=1
```

Keep optional expensive stages disabled for the first pilot:

```bash
export OPTIONAL_RUN_OPENMVS=0
export OPTIONAL_RUN_ODM=0
export OPTIONAL_RUN_NERFSTUDIO=0
```

For the full one-shot run, use the wrapper command below. It requests the optional OpenMVS/ODM/Nerfstudio stages by passing `-EnableFullOptionalMethods`, which sets `OPTIONAL_RUN_OPENMVS=1`, `OPTIONAL_RUN_ODM=1`, `OPTIONAL_RUN_NERFSTUDIO=1`, and `SKIP_RESEARCH_METHODS=0` on the pod. These stages are best-effort and should report soft failures rather than blocking the VGGT/COLMAP/R3/LingBot return if the selected pod image lacks an external dependency.

Only set these after the exact VGGeTR/VG2GT repo/checkpoint/API is confirmed:

```bash
# export VGGETR_COMMAND_TEMPLATE="python /workspace/vggetr/run.py --images {raw_images} --out {output_dir}"
# export VG2GT_COMMAND_TEMPLATE="python /workspace/vg2gt/run.py --images {raw_images} --out {output_dir}"
```

## HF Dataset Preflight

After the pod starts:

```bash
python - <<'PY'
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is not set")

info = HfApi(token=token).repo_info("Grimster/FPV_Hezbo", repo_type="dataset")
print("dataset:", info.id)
print("private:", info.private)
print("sha:", info.sha)
PY
```

This checks access without downloading the whole dataset.
## Local Prelaunch Audit

Before creating or spending time on a pod, run status plus both local gates:

```powershell
fpv runpod status-4090-launch
fpv runpod smoke-package --source outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip
fpv runpod audit-4090-launch
.\outputs\h100\full_161_run\launch\PRELAUNCH_AUDIT_4090_HF.ps1
```

The `fpv` command is source-owned and validates the 4090/HF manifest, helper
paths, package contract, expanded-method runner, package size, and SHA. The
PowerShell helper is the operator checklist that also checks ZIP duplicate
entries and pod-generated return instructions. Use `--fast` / `-Fast` only for
quick iteration; use the full checks before paid upload.


## R3 And LingBot-Map Optional Research Stages

Additional launch helper:

```text
outputs/h100/full_161_run/launch/RUNPOD_INSTALL_TRAIN_R3_LINGBOT_4090.sh
```

Use `-UseR3LingBotRunner` with `UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1` to upload and start this helper. It installs the official R3 relative-regression repository and LingBot-Map streaming reconstruction repository on the pod, then writes optional reports into the expanded run folder:

- `r3_reconstruction_manifest.json`
- `r3_training_report.json`
- `r3_dataset_prep_report.json`
- `r3_outputs.zip`
- `lingbot_map_manifest.json`
- `lingbot_map_outputs.zip`

R3 training is not treated as VGGT fine-tuning. It is an optional external bounded training/smoke path. The helper can prepare a pseudo CUT3R-style `R3_DATA_ROOT` bridge from local frame packs and writes `r3_dataset_prep_report.json`; placeholder depth/pose/camera/intrinsic files are smoke-test scaffolding, not metric supervision.

LingBot-Map is treated as inference/reconstruction first. Outputs stay relative and local-only; do not label them as geolocation, map projection, route, approach, speed, standoff, or guidance. The R3/LingBot wrapper auto-resolves a direct image-sequence directory from nested frame packs and uses Python `zipfile` for optional return packaging, avoiding dependency on `zip`/`unzip` in minimal RunPod images.

## Additional Optional Slots Worth Tracking

The launch manifest also records optional slots for HLOC, GLOMAP, Depth Anything V2, Metric3D, Viser, and SuperSplat. These are not hard dependencies for the first 4090 run. Treat HLOC/GLOMAP as sparse-reconstruction alternatives, Depth Anything V2/Metric3D as diagnostic priors without metric claims, Viser as a local viewer aid, and SuperSplat as manual post-processing after splat export.
## Pilot Run

From this workstation, the shortest upload path is below. The upload helper automatically runs `PRELAUNCH_AUDIT_4090_HF.ps1 -Fast` before any SSH/SCP work:

```powershell
.\outputs\h100\full_161_run\launch\UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1 -HostName <runpod_ip> -Port <tcp_port> -IdentityFile $env:USERPROFILE\.ssh\id_ed25519
```

Add `-Start` to start the default 3-item scout pilot in the background after upload and hash verification. Use `-Foreground` only for deliberate remote stream debugging. Use `-SkipPrelaunchAudit` only for deliberate audit-helper troubleshooting. Use `-RunMode full_packaged` or `-RunMode full_hf_rebuild` for later full runs.

For the shortest pod-side path, copy/paste `outputs/h100/full_161_run/launch/RUNPOD_RUN_4090_HF.sh` after uploading the package and setting `HF_TOKEN`. The default `FPV_RUN_MODE=pilot_hf_scout3` runs a 3-item scout pilot; later use `full_packaged` or `full_hf_rebuild`.

Upload:

```text
outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip
```

Then on the pod:

```bash
set -e
mkdir -p /workspace/fpv-expanded
cd /workspace/fpv-expanded
echo "1d36a3968b6dc64664265120e85489d120ede5b1e33032d7088047bbe97c232d  /workspace/runpod_job_4090_hf_full_stack_clean.zip" | sha256sum -c -
python -m zipfile -e /workspace/runpod_job_4090_hf_full_stack_clean.zip .
python scripts/05_expanded_runtime_check.py
python scripts/06_hf_dataset_preflight.py
python scripts/07_hf_dataset_frame_packs.py
cat expanded_runtime_check.json
cat hf_dataset_report.json
cat hf_frame_pack_report.json
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("expanded_runtime_check.json").read_text(encoding="utf-8"))
status = report.get("readiness_status")
if status not in {"ready_full_stack", "ready_4090_pilot"}:
    print(f"STOP: readiness_status={status}")
    for action in report.get("next_actions", []):
        print(f"- {action}")
    raise SystemExit(2)
print(f"preflight gate: {status}")
PY
bash run_all.sh
```

This package contains the local prepared full 161-clip frame packs by default, but the recommended 4090 pilot rebuilds a small scout-only manifest from the HF snapshot so `HF_DATASET_MAX_ITEMS=3` and `HF_DATASET_FRAME_PACK_TIERS=scout` actually limit the run. After the pilot passes, either set `HF_DATASET_BUILD_FRAME_PACKS=0` to run the packaged full manifest, or keep `HF_DATASET_BUILD_FRAME_PACKS=1` and remove the max-items limit while using `HF_DATASET_FRAME_PACK_TIERS=scout,main,high_detail` for a full HF-derived rebuild. The return contract is unchanged and includes `hf_dataset_report.json` and `hf_frame_pack_report.json` when those stages run.

## Monitor

While the pod is running:

```powershell
.\outputs\h100\full_161_run\launch\MONITOR_RUNPOD_4090_HF.ps1 -HostName <runpod_ip> -Port <tcp_port> -IdentityFile $env:USERPROFILE\.ssh\id_ed25519
```

Add `-Follow` to refresh every 20 seconds.
Status without continuous follow:

```powershell
.\outputs\h100\full_161_run\launch\CONTROL_RUNPOD_4090_HF.ps1 -HostName <runpod_ip> -Port <tcp_port> -IdentityFile $env:USERPROFILE\.ssh\id_ed25519
```

Stop a detached run only when you intentionally want to end it:

```powershell
.\outputs\h100\full_161_run\launch\CONTROL_RUNPOD_4090_HF.ps1 -HostName <runpod_ip> -Port <tcp_port> -IdentityFile $env:USERPROFILE\.ssh\id_ed25519 -Action stop
```

Use `-ForceKill` only if a normal stop does not end the remote process.

## Download And Validate

Download:

```text
/workspace/fpv-expanded/h100_return.zip
```

Then either download and validate in one step:

```powershell
.\outputs\h100\full_161_run\launch\DOWNLOAD_AND_VALIDATE_RETURN.ps1 -HostName <runpod_ip> -Port <tcp_port> -IdentityFile $env:USERPROFILE\.ssh\id_ed25519
```

or validate an already downloaded return ZIP:

```powershell
.\outputs\h100\full_161_run\launch\VALIDATE_RETURN.ps1 -ReturnZip <local_return_zip>
```

The validation helper runs:

- `fpv h100 inspect-return`
- `fpv h100 inspect-return --json`
- `fpv h100 postflight-return`
- `fpv h100 optional-report`
- `fpv h100 import-return --dry-run`
- `fpv runpod final-audit-4090-return --source <local_return_zip>`

## Completion Rule

This run is not done until a fresh return package is validated locally. A local
package, launch manifest, or old compact-bundle return is not enough.

## Safety Boundary

No geolocation. No map projection. No meters. No true speed/standoff/dive-angle
claims. No route, approach, launch, target-coordinate, guidance, or
next-maneuver inference.
