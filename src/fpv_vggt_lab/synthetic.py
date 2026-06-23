from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_synthetic_video(
    output: Path,
    frames: int = 48,
    width: int = 320,
    height: int = 180,
    fps: float = 12.0,
) -> Path:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    try:
        for index in range(frames):
            frame = _synthetic_frame(index=index, total=frames, width=width, height=height)
            writer.write(frame)
    finally:
        writer.release()

    return output


def _synthetic_frame(index: int, total: int, width: int, height: int) -> np.ndarray:
    x_grad = np.linspace(20, 160, width, dtype=np.uint8)
    y_grad = np.linspace(30, 120, height, dtype=np.uint8)[:, None]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0] = x_grad
    base[..., 1] = y_grad
    base[..., 2] = ((x_grad[None, :] // 2) + (y_grad // 2)).astype(np.uint8)

    progress = index / max(total - 1, 1)
    cx = int(20 + progress * max(width - 40, 1))
    cy = int(height * (0.45 + 0.2 * np.sin(progress * np.pi * 2)))
    radius = max(6, min(width, height) // 12)

    cv2.circle(base, (cx, cy), radius, (235, 235, 235), thickness=-1)
    cv2.rectangle(
        base,
        (max(0, width - cx - radius), max(0, height // 5)),
        (min(width - 1, width - cx + radius), min(height - 1, height // 5 + radius)),
        (40, 220, 180),
        thickness=-1,
    )
    cv2.line(
        base,
        (0, int(height * 0.8)),
        (width - 1, int(height * (0.72 + 0.05 * np.cos(progress * np.pi * 2)))),
        (240, 180, 60),
        thickness=2,
    )
    cv2.putText(
        base,
        f"F{index:03d}",
        (8, max(18, height - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return base

