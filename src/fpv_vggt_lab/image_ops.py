from __future__ import annotations

import cv2
import numpy as np


def apply_overlay_masks(frame: np.ndarray, mask_regions: list[list[float]] | None) -> np.ndarray:
    if not mask_regions:
        return frame
    output = frame.copy()
    height, width = output.shape[:2]
    fill = np.median(output.reshape(-1, 3), axis=0).astype(np.uint8)
    for region in mask_regions:
        if len(region) != 4:
            continue
        x0 = max(0, min(width, int(round(float(region[0]) * width))))
        y0 = max(0, min(height, int(round(float(region[1]) * height))))
        x1 = max(0, min(width, int(round(float(region[2]) * width))))
        y1 = max(0, min(height, int(round(float(region[3]) * height))))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = output[y0:y1, x0:x1]
        patch[:] = fill
        output[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (0, 0), 3.0)
    return output
