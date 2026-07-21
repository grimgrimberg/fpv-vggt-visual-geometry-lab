# ADR-005: Authorized derived point-cloud publication

## Status

Accepted on 2026-07-11.

## Context

The project owner reported on 2026-07-11 that dataset maintainer Itamar approved
publication of one derived VGGT Omega browser point sample. This direct approval is
a narrow exception to the repository's default rule that real-video-derived point
clouds remain local-only.

Publication through Git and GitHub Pages creates durable history. Removing a file
from the current site later does not erase it from existing clones, caches, or prior
Git commits, so the exception must remain explicit, reviewable, and fail closed.

## Decision

The public builder may publish the following two assets only when an operator sets
the explicit point-cloud opt-in and supplies both authorization provenance and
attribution:

- `points_preview.bin`, copied byte-for-byte as
  `scenes/<slug>/geometry/vggt_omega_points.f32.bin`;
- `points_preview_colors.bin`, copied byte-for-byte as
  `scenes/<slug>/geometry/vggt_omega_colors.rgb8.bin`.

The published coordinates remain `relative_only`. The browser applies a display
similarity transform derived from the raw relative camera path; the builder does
not alter the authorized source-coordinate sample bytes.

The scene contract records both browser-sample and full-source point counts, byte
counts, data types, component counts, SHA-256 hashes, backend, display transform,
authorization provenance, and attribution. The public-file audit accepts only the
two exact generated binary paths declared by that build. The default build remains
aggregate-only and rejects every binary file.

## Exclusions

This approval does not authorize publication of original video, extracted frames,
thumbnails, recognizable reprojections, full PLY or GLB geometry, raw NPZ or other
backend arrays, other point-cloud samples or scenes, absolute machine paths, or any
new operational, geolocation, guidance, targeting, or metric-scale claim.

## Consequences

The exception is deliberately opt-in and attributable. A build without complete
authorization metadata cannot publish the point sample. Full backend arrays remain
withheld, and the public viewer must state that coordinates are scale ambiguous and
relative rather than metric.
