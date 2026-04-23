"""Smoke tests for depth-layer assignment + per-layer mesh build."""
from __future__ import annotations

import numpy as np
import pytest

from monoworld.geometry import (
    build_per_layer_meshes,
    build_triangle_mesh_from_depth,
    intrinsics_from_fov,
)
from monoworld.segmentation import (
    assign_depth_layers,
    colorize_layer_mask,
    layer_color_legend,
)


def test_quantile_layers_partition_image():
    """Three quantile layers should each cover ~1/3 of valid pixels."""
    H, W = 30, 30
    # Linear depth ramp from 0.5 to 5.0
    depth = np.tile(np.linspace(0.5, 5.0, W, dtype=np.float32), (H, 1))
    layers = assign_depth_layers(depth, num_layers=3, method="quantile")
    counts = np.array([(layers == k).sum() for k in range(3)])
    total = layers.size
    # Each layer should be roughly H*W/3 (within 5% slack).
    for c in counts:
        assert abs(c - total / 3) < 0.05 * total


def test_layer_0_is_closest():
    H, W = 10, 10
    depth = np.zeros((H, W), dtype=np.float32)
    depth[:, :W // 2] = 1.0   # left half: close
    depth[:, W // 2:] = 5.0   # right half: far
    layers = assign_depth_layers(depth, num_layers=2, method="quantile")
    # Layer 0 should overlap with the close (left) half.
    assert layers[0, 0] == 0
    assert layers[0, W - 1] == 1


def test_invalid_pixels_marked_neg1():
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    depth[0, 0] = 0.0  # below min_depth
    layers = assign_depth_layers(depth, num_layers=3, method="quantile",
                                 min_depth=0.1)
    assert layers[0, 0] == -1
    assert (layers != -1).sum() == 24


def test_kmeans_runs_and_orders_centers():
    rng = np.random.default_rng(0)
    # Three clusters at depth 1, 3, 5.
    depth = np.concatenate([
        rng.normal(1.0, 0.05, 100),
        rng.normal(3.0, 0.05, 100),
        rng.normal(5.0, 0.05, 100),
    ]).reshape(15, 20).astype(np.float32)
    layers = assign_depth_layers(depth, num_layers=3, method="kmeans")
    # Closest cluster should be labeled 0.
    closest_idx = np.argmin(depth.flatten())
    assert layers.flatten()[closest_idx] == 0


def test_colorize_returns_correct_shape():
    layers = np.zeros((4, 6), dtype=np.int8)
    layers[1] = 1
    layers[2] = 2
    layers[3] = -1
    img = colorize_layer_mask(layers, num_layers=3)
    assert img.shape == (4, 6, 3)
    assert img.dtype == np.uint8
    # Invalid row should be black.
    assert (img[3] == 0).all()
    # Each layer row should have nonzero color.
    assert (img[0] > 0).any()


def test_layer_color_legend_size():
    legend = layer_color_legend(3)
    assert set(legend.keys()) == {0, 1, 2}
    for v in legend.values():
        assert len(v) == 3


def test_per_layer_meshes_partition_triangles():
    """The total triangle count across layers should be <= the unlayered count
    (per-layer meshes drop triangles that span layer boundaries)."""
    H, W = 30, 30
    depth = np.tile(np.linspace(0.5, 5.0, W, dtype=np.float32), (H, 1))
    K = intrinsics_from_fov(W, H, fov_deg=55.0)
    layer_mask = assign_depth_layers(depth, num_layers=3, method="quantile")

    # Unlayered baseline (no edge culling for a clean comparison).
    _, tris_full, _ = build_triangle_mesh_from_depth(
        depth, K, stride=1, edge_threshold_factor=0,
    )
    per_layer = build_per_layer_meshes(
        depth, layer_mask, K, stride=1, edge_threshold_factor=0,
    )
    total = sum(t.shape[0] for _, t, _ in per_layer.values())
    # Per-layer meshes only include cells where ALL 4 corners are in the same
    # layer, so total triangles should be strictly less.
    assert total <= tris_full.shape[0]
    # And each layer should have at least some triangles.
    assert all(t.shape[0] > 0 for _, t, _ in per_layer.values())
    # Three layers expected.
    assert set(per_layer.keys()) == {0, 1, 2}
