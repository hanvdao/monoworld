"""Smoke tests for geometry math."""
from __future__ import annotations

import numpy as np
import pytest

from monoworld.geometry import (
    Intrinsics,
    depth_to_colored_pointcloud,
    intrinsics_from_fov,
    unproject_depth_to_points,
)


def test_intrinsics_center_and_fx():
    K = intrinsics_from_fov(width=640, height=480, fov_deg=60.0)
    assert K.cx == pytest.approx(320.0)
    assert K.cy == pytest.approx(240.0)
    # fx = (W/2) / tan(fov/2). For fov=60deg, tan(30) ~ 0.5774.
    assert K.fx == pytest.approx(320.0 / np.tan(np.radians(30)), rel=1e-4)
    assert K.fy == K.fx


def test_unproject_center_pixel_is_on_optical_axis():
    K = intrinsics_from_fov(64, 48, fov_deg=60.0)
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    points, valid = unproject_depth_to_points(depth, K, stride=1)
    assert valid.all()
    # Center pixel ~ (32, 24). X/Y should be ~0, Z = 2.
    center_idx = 24 * 64 + 32
    X, Y, Z = points[center_idx]
    assert abs(X) < 1e-3
    assert abs(Y) < 1e-3
    assert Z == pytest.approx(2.0)


def test_unproject_drops_out_of_range():
    K = intrinsics_from_fov(10, 10, fov_deg=55.0)
    depth = np.full((10, 10), 0.01, dtype=np.float32)  # below min_depth
    points, valid = unproject_depth_to_points(depth, K, min_depth=0.1, max_depth=50.0)
    assert points.shape[0] == 0
    assert not valid.any()


def test_colored_pointcloud_shape_agreement():
    image = (np.random.rand(20, 30, 3) * 255).astype(np.uint8)
    depth = np.full((20, 30), 1.5, dtype=np.float32)
    K = intrinsics_from_fov(30, 20, fov_deg=55.0)
    pcd = depth_to_colored_pointcloud(image, depth, K, stride=1)
    assert len(pcd.points) == 20 * 30
    assert len(pcd.colors) == 20 * 30
