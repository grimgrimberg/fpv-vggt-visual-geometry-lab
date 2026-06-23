# ADR-004: Data Storage And Media Non-Redistribution

Date: 2026-06-22

## Status

Accepted

## Context

The dataset repository licenses its metadata, manifests, and docs as CC0, but
the referenced videos and thumbnails are third-party media with no license grant
from that repository.

## Decision

Pipeline tables use durable local metadata formats such as Parquet or JSONL.
Videos, thumbnails, extracted real frames, real-video VGGT bundles, point
clouds, rendered clips, and review HTML embedding real media are local-only and
ignored by git.

## Consequences

The repository can contain code, synthetic tests, schemas, documentation, and
catalog references without redistributing third-party media.
