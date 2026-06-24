from __future__ import annotations


FORBIDDEN_FEATURE_TERMS = [
    "altitude",
    "approach",
    "coordinate",
    "coordinates",
    "dive_angle",
    "geo",
    "geocode",
    "geolocation",
    "latitude",
    "launch",
    "longitude",
    "maneuver",
    "map_projection",
    "mgrs",
    "route",
    "speed_mps",
    "standoff",
    "target",
    "utm",
]


def forbidden_feature_issues(columns: list[str]) -> list[str]:
    issues: list[str] = []
    for column in columns:
        normalized = column.lower()
        for term in FORBIDDEN_FEATURE_TERMS:
            if term in normalized:
                issues.append(
                    f"forbidden feature column {column!r} matches unsafe term {term!r}"
                )
                break
    return issues


def validate_feature_columns(columns: list[str]) -> None:
    issues = forbidden_feature_issues(columns)
    if issues:
        raise ValueError("; ".join(issues))
