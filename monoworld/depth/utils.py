"""Depth post-processing: disparity->depth, visualization, device resolution."""
from __future__ import annotations

import cv2
import numpy as np
import torch


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' -> 'cuda' if available, else 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def disparity_to_depth(
    disparity: np.ndarray,
    near: float = 0.5,
    far: float = 20.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert a relative disparity map to metric-ish depth.

    Args:
        disparity: HxW float array. Higher = closer (MiDaS convention).
        near: depth assigned to the closest pixel.
        far: depth assigned to the farthest pixel.

    Returns:
        HxW float32 depth map in [near, far].
    """
    d = disparity.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min < eps:
        # Degenerate case: flat prediction. Return mid-range.
        return np.full_like(d, 0.5 * (near + far), dtype=np.float32)
    d_norm = (d - d_min) / (d_max - d_min)        # [0,1], closer=1
    z = near + (1.0 - d_norm) * (far - near)      # [near, far]
    return z.astype(np.float32)


def normalize_depth_for_vis(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to uint8 for visualization (closer = brighter)."""
    d = depth.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min < 1e-6:
        return np.zeros_like(d, dtype=np.uint8)
    # Invert so closer objects render bright.
    vis = 1.0 - (d - d_min) / (d_max - d_min)
    return (vis * 255.0).clip(0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray, cmap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
    """Return an HxWx3 uint8 BGR colorized depth image."""
    gray = normalize_depth_for_vis(depth)
    return cv2.applyColorMap(gray, cmap)
