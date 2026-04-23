"""Synthesize pinhole intrinsics from image size + horizontal FOV."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


def intrinsics_from_fov(width: int, height: int, fov_deg: float = 55.0) -> Intrinsics:
    """Construct pinhole intrinsics assuming square pixels + centered principal point."""
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
