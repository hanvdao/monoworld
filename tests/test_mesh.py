"""Smoke tests for mesh construction and edge culling."""
from __future__ import annotations

import numpy as np
import pytest

from monoworld.geometry import (
    auto_stride_for_budget,
    build_triangle_mesh_from_depth,
    intrinsics_from_fov,
)


def test_auto_stride_hits_budget():
    # 1280x720 image, target 200k tris -> stride ~3
    s = auto_stride_for_budget(1280, 720, target_triangles=200_000)
    assert 2 <= s <= 4
    # Tiny image -> stride 1
    s = auto_stride_for_budget(100, 100, target_triangles=200_000)
    assert s == 1


def test_mesh_has_expected_topology():
    """For a flat depth map at stride 1, every grid cell should produce 2 tris."""
    H, W = 10, 12
    depth = np.full((H, W), 1.0, dtype=np.float32)
    K = intrinsics_from_fov(W, H, fov_deg=55.0)
    verts, tris, uvs = build_triangle_mesh_from_depth(
        depth, K, stride=1, edge_threshold_factor=0,  # no culling
    )
    # All HxW vertices kept, but triangles are 2 * (H-1) * (W-1).
    assert verts.shape == (H * W, 3)
    assert uvs.shape == (H * W, 2)
    assert tris.shape == (2 * (H - 1) * (W - 1), 3)
    # Triangle indices must be in range.
    assert tris.min() >= 0
    assert tris.max() < verts.shape[0]


def test_edge_culling_removes_huge_triangles():
    """A depth jump should produce big triangles that get culled."""
    H, W = 20, 20
    depth = np.full((H, W), 1.0, dtype=np.float32)
    # Right half is far away — every triangle bridging the seam will stretch.
    depth[:, W // 2:] = 10.0
    K = intrinsics_from_fov(W, H, fov_deg=55.0)

    _, tris_no_cull, _ = build_triangle_mesh_from_depth(
        depth, K, stride=1, edge_threshold_factor=0,
    )
    _, tris_cull, _ = build_triangle_mesh_from_depth(
        depth, K, stride=1, edge_threshold_factor=2.0,
    )
    # Culling should drop a meaningful fraction of triangles.
    assert tris_cull.shape[0] < tris_no_cull.shape[0]
    # But not all of them — most cells are flat.
    assert tris_cull.shape[0] > tris_no_cull.shape[0] * 0.5


def test_invalid_depth_excludes_cells():
    H, W = 10, 10
    depth = np.full((H, W), 1.0, dtype=np.float32)
    depth[0, 0] = 0.0  # below min_depth
    K = intrinsics_from_fov(W, H, fov_deg=55.0)
    _, tris, _ = build_triangle_mesh_from_depth(
        depth, K, stride=1, min_depth=0.1, edge_threshold_factor=0,
    )
    # Cell (0,0) should be excluded -> 2 fewer triangles than the flat case.
    expected_full = 2 * (H - 1) * (W - 1)
    assert tris.shape[0] == expected_full - 2


def test_uvs_in_unit_range():
    H, W = 16, 24
    depth = np.full((H, W), 2.0, dtype=np.float32)
    K = intrinsics_from_fov(W, H, fov_deg=55.0)
    _, _, uvs = build_triangle_mesh_from_depth(depth, K, stride=1)
    assert uvs.min() >= 0.0
    assert uvs.max() <= 1.0
