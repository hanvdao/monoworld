"""Depth-based scene layering.

Splits a depth map into K discrete layers (foreground -> background) so that
downstream stages (per-layer meshing, disocclusion masking, inpainting) can
reason about them independently.

Two methods supported:
    - "quantile": bin pixels by depth quantiles. Deterministic, fast, robust
                  to outliers. Default.
    - "kmeans":   1D k-means on depth. Adapts to depth distribution but can
                  produce unbalanced layers.

Layer 0 is always the closest layer.
"""
from __future__ import annotations

import numpy as np


# Distinct, perceptually separable colors for layer visualization.
# Up to 6 layers; project plan only ever uses 3.
_LAYER_COLORS_BGR = np.array([
    (255,  80,  80),   # layer 0 (closest) — bright blue
    ( 80, 220, 255),   # layer 1            — yellow
    ( 80, 255, 120),   # layer 2 (farthest) — green
    (180,  80, 255),   # layer 3            — magenta
    (255, 200,  80),   # layer 4            — cyan
    (180, 180, 180),   # layer 5            — grey
], dtype=np.uint8)


def assign_depth_layers(
    depth: np.ndarray,
    num_layers: int = 3,
    method: str = "quantile",
    min_depth: float = 0.1,
    max_depth: float = 50.0,
) -> np.ndarray:
    """Assign each pixel to a discrete depth layer in [0, num_layers).

    Args:
        depth: HxW float depth map.
        num_layers: number of layers (typically 3: foreground/mid/background).
        method: "quantile" or "kmeans".
        min_depth, max_depth: pixels outside this range are marked invalid (-1).

    Returns:
        HxW int8 array. Values in [0, num_layers) are layer IDs (0 = closest).
        Value -1 marks invalid pixels.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    if num_layers < 1 or num_layers > len(_LAYER_COLORS_BGR):
        raise ValueError(f"num_layers must be in [1, {len(_LAYER_COLORS_BGR)}]")

    H, W = depth.shape
    layers = np.full((H, W), -1, dtype=np.int8)
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if not valid.any():
        return layers

    valid_depths = depth[valid]

    if method == "quantile":
        # Quantile boundaries within the valid depth range. linspace(0, 1, n+1)[1:-1]
        # gives the n-1 internal cut points.
        quantiles = np.linspace(0.0, 1.0, num_layers + 1)[1:-1]
        cuts = np.quantile(valid_depths, quantiles)
        # np.digitize: returns bin indices in [0, num_layers).
        bin_ids = np.digitize(depth, cuts)  # 0..num_layers
        bin_ids = np.clip(bin_ids, 0, num_layers - 1)
        layers[valid] = bin_ids[valid].astype(np.int8)

    elif method == "kmeans":
        # Simple Lloyd's k-means in 1D. Init at quantile centers for stability.
        centers = np.quantile(
            valid_depths,
            np.linspace(0.5 / num_layers, 1.0 - 0.5 / num_layers, num_layers),
        )
        x = valid_depths.reshape(-1, 1)
        for _ in range(20):
            # Assign each point to nearest center.
            dists = np.abs(x - centers.reshape(1, -1))
            assignments = np.argmin(dists, axis=1)
            new_centers = np.array([
                x[assignments == k].mean() if (assignments == k).any() else centers[k]
                for k in range(num_layers)
            ])
            if np.allclose(new_centers, centers, atol=1e-4):
                break
            centers = new_centers
        # Sort centers ascending so layer 0 is closest.
        order = np.argsort(centers)
        remap = np.argsort(order)  # invert
        assignments = remap[assignments]
        layers[valid] = assignments.astype(np.int8)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'quantile' or 'kmeans'.")

    return layers


def colorize_layer_mask(layers: np.ndarray, num_layers: int) -> np.ndarray:
    """Return HxWx3 uint8 BGR image color-coding the layer assignment.

    Invalid pixels (-1) are rendered black.
    """
    H, W = layers.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for k in range(num_layers):
        out[layers == k] = _LAYER_COLORS_BGR[k]
    return out


def layer_color_legend(num_layers: int) -> dict[int, tuple[int, int, int]]:
    """Return {layer_id: (B, G, R)} mapping for documentation / metadata."""
    return {k: tuple(int(c) for c in _LAYER_COLORS_BGR[k]) for k in range(num_layers)}
