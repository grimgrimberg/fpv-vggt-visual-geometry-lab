# ADR-003: No Geolocation Or Operational Features

Date: 2026-06-22

## Status

Accepted

## Context

Relative camera paths and dataset descriptions can be tempting to combine with
maps, target labels, or tactical interpretation. That would move the project
outside the intended historical-media analysis scope.

## Decision

The project forbids geolocation, map-based approach-corridor analysis,
launch-point inference, target-coordinate inference, guidance, route
optimization, next-maneuver prediction, and tactical recommendations.

Dataset descriptions remain source metadata only.

## Consequences

Outputs must emphasize ambiguity: no meters, no coordinates, no target-relative
claims, and no operational ranking.
