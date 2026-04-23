"""Unproject a depth map to a colored 3D point cloud.

Coordinate convention: vertices are returned in **Three.js convention**:
    +X right, +Y up, +Z forward.
Achieved by negating Y after backprojection from OpenCV camera coords.
The viewer renders the .ply directly with no flip.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from .intrinsics import Intrinsics


def unproject_depth_to_points(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject a depth map to 3D points (Three.js coords)."""
    H, W = depth.shape
    us = np.arange(0, W, stride, dtype=np.float32)
    vs = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    z = depth[::stride, ::stride].astype(np.float32)
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)

    X = (uu - intrinsics.cx) * z / intrinsics.fx
    Y = (vv - intrinsics.cy) * z / intrinsics.fy
    Z = z
    Y = -Y  # OpenCV +Y down -> Three.js +Y up

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    valid_flat = valid.reshape(-1)
    return points[valid_flat], valid_flat


def depth_to_colored_pointcloud(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
) -> o3d.geometry.PointCloud:
    """Build an Open3D colored point cloud from an image + depth.

    Shapes must agree: image_rgb is HxWx3 uint8, depth is HxW float.
    """
    if image_rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"Image/depth shape mismatch: {image_rgb.shape[:2]} vs {depth.shape}"
        )
    points, valid_flat = unproject_depth_to_points(
        depth, intrinsics, stride=stride,
        min_depth=min_depth, max_depth=max_depth,
    )

    colors = image_rgb[::stride, ::stride].reshape(-1, 3).astype(np.float32) / 255.0
    colors = colors[valid_flat]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd
