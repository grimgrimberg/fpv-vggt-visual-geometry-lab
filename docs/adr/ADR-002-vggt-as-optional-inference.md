# ADR-002: VGGT As Optional Inference

Date: 2026-06-22

## Status

Accepted

## Context

The local development machine has limited RAM and no required local GPU path.
VGGT execution can also happen through Hugging Face, Colab, RunPod, or a local
installation, and output shapes may vary.

## Decision

V1 is import-first. The repository owns a stable VGGT prediction bundle format,
and all VGGT sources convert into that format. Core tests use mocked bundles and
must not require VGGT, CUDA, Hugging Face, Colab, or RunPod.

## Consequences

The core pipeline remains testable on ordinary hardware. Real VGGT inference can
be added later without coupling the rest of the project to one runtime.
