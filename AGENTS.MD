# AGENTS.md

## Project identity

This repository is `fpv-vggt-visual-geometry-lab`.

It is an offline, retrospective academic research project for visual-geometry
reconstruction, GNC-style trajectory analysis, and ML benchmarking on already
published or otherwise historical recorded FPV footage. It is post-flight only:
no real-time processing, live guidance, deployment, or operational decision
support belongs in this repository.

The core goal is to:
1. ingest pinned dataset catalogs or external media indexes,
2. download a local/RunPod processing cache of referenced media,
3. isolate FPV-flight segments from edited clips with automated diagnostics,
4. extract frames and calibration metadata,
5. run or import VGGT and other 3D reconstruction predictions,
6. reconstruct camera poses, point clouds, and relative/metric trajectory states,
7. visualize relative camera paths and reconstruction quality,
8. score reconstruction reliability and artifact contamination,
9. estimate scale-aware historical velocity, acceleration, and GNC descriptors
   when documented structural priors support metric scaling,
10. optionally train or fine-tune offline models for reconstruction quality,
    trajectory representation, time-to-visible-terminal-event, and retrospective
    system-identification research,
11. export local-only review artifacts and robotics telemetry logs for offline
    playback and academic analysis.

## Safety scope

Allowed:
- offline video ingestion
- local and RunPod media caching
- Hugging Face dataset preflight and downloads
- S3 or CloudFront catalog parsing and media downloads into ignored local caches
- frame extraction
- automated segment annotation and quality diagnostics
- optical flow
- VGGT inference/import/fine-tuning when the environment supports it
- COLMAP, OpenDroneMap (ODM), MASt3R, DUSt3R, VGGeTR, Nerfstudio, R3 3D
  reconstruction, Lingbot map, and related 3D reconstruction tooling
- relative camera-pose visualization
- point-cloud visualization
- reconstruction diagnostics
- scale-free and scale-aware historical path descriptors
- retrospective metric scale estimation from documented structural priors
- Kalman filtering, EKF, UKF, particle filtering, B-splines, Gaussian-process
  smoothers, robust losses, and continuous-time pose or trajectory fitting
- passive post-flight system identification, including INDI-style analysis,
  batch least-squares optimization, inverse dynamics, and historical PID or
  control-loop trend estimation
- time-to-visible-terminal-event modeling as retrospective benchmarking on
  historical edited footage
- dashboard and local HTML review artifacts for human inspection
- local robotics telemetry exports for offline playback and analysis
- reconstruction-suitability shortlisting using non-operational quality signals

Not allowed:
- live detection or live tracking
- operational target detection, target selection, or target vulnerability analysis
- deployment of guidance/control outputs or route optimization
- real-time maneuver recommendations or next operational action recommendations
- precise geolocation, launch-point inference, target-coordinate inference, or
  map-based approach-corridor analysis
- attack simulation, weapons-effectiveness analysis, or tactical recommendations
- redistribution of third-party media or real-media-derived artifacts that should
  remain local-only

Dataset descriptions can be retained as source metadata and converted into
non-operational analytic labels for offline research. Do not convert source
metadata into target selection, target vulnerability, geolocation, attack
planning, or tactical decision-support outputs.

## Terminology

Use:
- "relative camera path"
- "VGGT coordinate frame"
- "scale-free path descriptor"
- "metric scale prior"
- "historical estimated velocity"
- "historical estimated acceleration"
- "trajectory representation"
- "time-to-visible-terminal-event"
- "TTVE"
- "reconstruction reliability"
- "post-flight system identification"
- "control-loop estimate"
- "maneuver prediction"

Terms such as "actual speed", "control", "maneuver prediction", "flight
dynamics", and "trajectory representation" are authorized when they describe
offline estimates from completed historical video clips. Attach uncertainty,
scale-prior provenance, and calibration assumptions when the terms imply metric
or physical quantities.

Do not present reconstructed trajectories, scale estimates, control-loop trends,
or maneuver predictions as live, tactical, deployable, or targeting outputs.

## Data rules

Never commit:
- videos
- thumbnails
- extracted frames
- dense optical flow
- VGGT point clouds from real videos
- COLMAP, ODM, Nerfstudio, MASt3R, DUSt3R, VGGeTR, R3, Lingbot, or other real
  video point clouds/reconstructions
- smoothed 6DoF state exports derived from real videos
- ROS `.bag`, `.mcap`, or structured CSV telemetry exports derived from real
  videos unless explicitly approved as a safe local-only artifact
- rendered clips from real videos unless explicitly intended as local-only output
- rendered HTML review pages that embed real frames or real point clouds
- screenshots of real-media review pages
- model checkpoints trained on real media unless explicitly approved for a safe
  release review

Ignored local paths:
- `data/media/`
- `data/frames/`
- `data/features/`
- `data/vggt/`
- `data/geometry/`
- `data/models/`
- `data/annotations/`
- `outputs/`

The dataset repo's metadata/catalog can be used, but referenced media are
third-party materials and must not be redistributed by this project.

Pin dataset catalog snapshots by commit, manifest date, or retrieval date. The
source dataset is expected to change over time.

For S3 or CloudFront-backed datasets such as `fpv-drone-strikes-lebanon-dataset`:
- treat the repository as an index unless it actually provides a native Hugging
  Face dataset implementation
- write ingestion scripts that parse repository metadata for stable source URLs,
  clip identifiers, retrieval timestamps, and optional checksums
- download referenced `.mp4` media directly into `data/media/` or a RunPod
  scratch/cache directory
- write a manifest containing source URL, retrieval time, local cache path,
  checksum when available, and processing status
- keep downloaded media and real-media-derived artifacts ignored and local-only

Standardized telemetry export is allowed for offline academic playback and
analysis. The pipeline may export smoothed 6DoF state-space data, reconstructed
point clouds, reconstruction reliability scores, and derived GNC metrics into ROS
`.bag`, `.mcap`, or structured CSV files for tools such as Foxglove Studio or
RViz. These exports must remain local/ignored unless the user explicitly asks for
a safe release review.

VTX and camera hardware profiling is allowed. The project may maintain hardcoded
camera intrinsic profiles for common FPV VTX/camera systems such as DJI O3,
Caddx Vista, and standard analog cameras to provide stronger priors for
Kannala-Brandt distortion models, rolling-shutter compensation, and camera
intrinsic estimation. Profiles must record source, assumptions, and uncertainty
when known.

## 3D reconstruction and GNC rules

Do not vendor large third-party reconstruction projects into this repository.
Prefer optional adapters, setup scripts, pinned environment files, or external
install steps for VGGT, COLMAP, ODM, MASt3R, DUSt3R, VGGeTR, Nerfstudio, R3 3D
reconstruction, Lingbot map, and similar tools.

Provide:
- an import path for externally generated reconstruction predictions
- a repo-owned VGGT/reconstruction prediction bundle format where practical
- adapters only when the corresponding tool is installed or prepared externally
- mocks or lightweight fixtures for fast local tests

RunPod, Hugging Face, Colab, or local GPU runs are valid ways to produce
importable bundles or full pipeline outputs. For GPU-heavy runs, aggressively pin
CUDA-sensitive dependencies such as PyTorch versions, CUDA build flags, xformers,
COLMAP-related packages, Nerfstudio dependencies, and MASt3R/DUSt3R environments
so setup is deterministic on RTX 4090-class machines.

Reconstruction and trajectory modules may use:
- robust camera intrinsic estimation for ultra-wide or fisheye lenses, including
  Kannala-Brandt-style models where supported
- hardcoded VTX/camera profiles as priors for camera intrinsics and rolling
  shutter compensation
- velocity-aware or continuous-time pose estimation for rolling-shutter skew
  during high-angular-rate motion
- robust losses such as Huber or Tukey biweight for spline fitting, pose
  smoothing, bundle adjustment, and trajectory diagnostics
- Kalman filters, EKF, UKF, particle filters, B-splines, Gaussian-process
  smoothers, and other probabilistic or continuous-time state estimators
- documented structural metric priors from bounding boxes, known static object
  dimensions, calibration targets, or other non-live references to resolve global
  scale ambiguity for offline velocity and acceleration metrics
- passive post-flight system identification, INDI-style analysis, batch
  least-squares optimization, inverse-dynamics analysis, and historical PID or
  control-loop trend estimation from completed video clips
- higher-order temporal derivatives such as jerk and snap when uncertainty and
  scale assumptions are recorded

Always document:
- global scale ambiguity and the chosen scale-prior provenance
- global rotation ambiguity
- pose outliers
- calibration uncertainty
- rolling-shutter and fisheye assumptions
- low-texture failures
- title-card/freeze/replay contamination
- edited-footage bias
- reconstruction reliability gates

Do not present GNC outputs as deployable autopilot commands, guidance logic,
route optimization, weapons-effectiveness analysis, or tactical recommendations.

## ML rules

Training and fine-tuning are allowed for offline retrospective research on
locally cached project data. This includes fine-tuning VGGT, training large video
foundation models, and training generative or continuous-time trajectory models
when the purpose is reconstruction quality, reliability estimation, artifact
rejection, trajectory representation, interpolation, smoothing, retrospective
TTVE benchmarking, or post-flight system-identification research.

Splits must be grouped by `video_id`, never random frame-row splits.

Always include:
- timestamp-only baseline
- reconstruction-reliability-stratified metrics
- warning when performance is dominated by clip-position leakage
- clear labeling that TTVE and trajectory predictions are retrospective estimates
  on historical edited footage, not live prediction or operational guidance

Keep generated models, checkpoints, and derived real-media artifacts local unless
the user explicitly asks for a safe release review.

## Testing

Use the smallest meaningful verification step for the change.

Fast local tests should prefer synthetic videos, mocks, and lightweight fixtures
so core checks do not require real media, CUDA, Hugging Face, Colab, or RunPod.

RunPod/full-stack checks may download real media, call VGGT or other heavy 3D
tools, require CUDA, use Hugging Face, and run against ignored local caches when
the task explicitly targets the full pipeline.

For rapid deployment work, minimal smoke testing is acceptable when it is paired
with concrete logs and manifests. Useful minimal checks include:
- `fpv --help`
- script syntax checks
- dependency/environment preflight
- catalog or manifest validation
- a small dry run or one-clip run when feasible

Use broader `pytest` coverage when editing shared library behavior, schemas,
import/export contracts, or safety-critical local-only artifact handling.

## V1 workflow rules

The v1 workflow is an automated single-script RunPod-capable pipeline:
1. ingest a pinned dataset catalog snapshot or external S3/CloudFront media index,
2. run dataset preflight and download selected real media into an ignored local or
   RunPod cache,
3. automatically propose and select FPV-flight segments using non-operational
   quality and reconstruction diagnostics,
4. sample frames deterministically,
5. run or import VGGT and optional heavy 3D reconstruction outputs,
6. compute reconstruction reliability, camera calibration quality, and artifact
   contamination diagnostics,
7. estimate relative and, when calibrated, metric 6DoF trajectory states,
8. run smoothing, Kalman filtering, robust spline fitting, and passive post-flight
   system-identification analyses where requested,
9. train or evaluate offline ML models when requested,
10. export local-only HTML review artifacts, manifests, telemetry logs, point
    clouds, and structured metrics.

The high-level `fpv run` command and RunPod `run_all` style scripts should
orchestrate the happy path without a mandatory human-in-the-loop gate. Human
review dashboards and local HTML artifacts are useful inspection outputs, not
required blockers for downstream automated analysis.

Each run must write:
- `run.log`
- `summary.json`
- `NEXT_STEPS.md`
- artifact paths
- stage statuses: `done`, `needs_data`, `needs_reconstruction_bundle`,
  `failed_soft`, or `blocked`
- a warning block: offline-only, no live guidance, no geolocation, no tactical
  recommendations, local-only media, reconstruction scale/rotation ambiguity,
  metric-prior provenance when used, and uncertainty notes
- an environment snapshot without secrets

Failure should produce useful review diagnostics where possible. Descriptor
extraction, comparison summaries, smoothing, metric scaling, telemetry export,
and system-identification outputs must record reconstruction reliability and
uncertainty so downstream analysis can filter weak reconstructions.

## Engineering style

Use Python 3.11+.
Use Typer for CLI.
Use Pydantic for schemas/config validation.
Use Parquet for intermediate tables.
Keep modules small and typed.
Prefer explicit warnings over silent assumptions.
