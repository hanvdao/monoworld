"""Disocclusion mask generation.

Given a set of depth layers, compute per-layer "disocclusion masks" — the
regions of each background layer that are hidden behind foreground layers in
the original view and would become visible if the camera translated slightly.

Strategy:
    For background layer k (k > 0):
        1. Take the union of all foreground layer masks (layers 0..k-1).
        2. Dilate that union by `dilation_px` pixels. The dilation simulates
           the fact that a small camera translation reveals a border region
           around every foreground occluder.
        3. The disocclusion mask for layer k = dilated_foreground AND layer_k_pixels.
           These are the pixels in layer k that sit right behind the edges of
           foreground objects.

The disocclusion mask is what the inpainter must fill: pixels that currently
have valid depth/color but whose *neighbors* are occluded, so a camera
translation would reveal unknown content at those borders.

For practical inpainting, we use a slightly different approach: we inpaint the
*entire image* at the foreground mask locations for each background layer.
This gives the inpainter a clean version of each layer with foreground removed.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_foreground_mask(
    layer_mask: np.ndarray,
    current_layer: int,
) -> np.ndarray:
    """Boolean mask of all pixels belonging to layers closer than `current_layer`."""
    return (layer_mask >= 0) & (layer_mask < current_layer)


def compute_disocclusion_mask(
    layer_mask: np.ndarray,
    current_layer: int,
    dilation_px: int = 15,
) -> np.ndarray:
    """Compute the disocclusion mask for a given background layer.

    Args:
        layer_mask: HxW int8 layer assignment (-1 = invalid, 0 = closest).
        current_layer: the layer we're computing disocclusion for (must be > 0).
        dilation_px: how many pixels to dilate foreground masks (simulates
            camera translation magnitude).

    Returns:
        HxW bool mask. True = this pixel in `current_layer` needs inpainting
        because it sits behind (or adjacent to) a foreground occluder.
    """
    if current_layer <= 0:
        return np.zeros(layer_mask.shape, dtype=bool)

    fg_mask = compute_foreground_mask(layer_mask, current_layer)
    fg_uint8 = fg_mask.astype(np.uint8) * 255

    # Dilate foreground to expand "shadow" into background.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1),
    )
    dilated = cv2.dilate(fg_uint8, kernel, iterations=1) > 0

    # The disocclusion region is where the dilated foreground overlaps with
    # the current layer OR invalid/foreground regions within the current layer
    # footprint. For inpainting, we want to fill wherever foreground objects
    # sit on top of this layer.
    is_current = layer_mask == current_layer
    inpaint_mask = fg_mask & ~is_current  # not needed, but safe

    # Actually, the most useful inpainting mask is: "everywhere that is NOT
    # this layer's own pixels, within the bounding area of this layer + some
    # margin." Simplest practical approach: inpaint at all foreground pixel
    # locations, so the background layer gets a clean version.
    return fg_mask | (dilated & ~is_current)


def compute_layer_inpaint_mask(
    layer_mask: np.ndarray,
    target_layer: int,
    dilation_px: int = 15,
) -> np.ndarray:
    """Compute the inpaint mask for a target background layer.

    The mask marks pixels that the inpainter should fill: all foreground
    pixels (layers 0..target_layer-1) plus a dilated border, restricted to
    the region where the target layer and its neighbors live.

    Returns:
        HxW uint8 mask (0 or 255) suitable for cv2.inpaint().
    """
    if target_layer <= 0:
        return np.zeros(layer_mask.shape, dtype=np.uint8)

    fg_mask = compute_foreground_mask(layer_mask, target_layer)
    fg_uint8 = fg_mask.astype(np.uint8) * 255

    # Dilate to cover the boundary region.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1),
    )
    dilated = cv2.dilate(fg_uint8, kernel, iterations=1)

    return dilated
