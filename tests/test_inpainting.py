"""Smoke tests for disocclusion mask generation and OpenCV inpainting."""
from __future__ import annotations

import numpy as np
import pytest

from monoworld.inpainting.disocclusion import (
    compute_foreground_mask,
    compute_layer_inpaint_mask,
)
from monoworld.inpainting.opencv_inpaint import inpaint_opencv
from monoworld.inpainting.pipeline import inpaint_all_layers


def test_foreground_mask_excludes_current_and_behind():
    layers = np.array([[0, 1, 2], [0, 1, -1]], dtype=np.int8)
    fg = compute_foreground_mask(layers, current_layer=2)
    expected = np.array([[True, True, False], [True, True, False]])
    np.testing.assert_array_equal(fg, expected)


def test_foreground_mask_layer0_is_empty():
    layers = np.zeros((3, 3), dtype=np.int8)
    fg = compute_foreground_mask(layers, current_layer=0)
    assert not fg.any()


def test_inpaint_mask_is_uint8_255():
    layers = np.zeros((10, 10), dtype=np.int8)
    layers[:, :5] = 0
    layers[:, 5:] = 1
    mask = compute_layer_inpaint_mask(layers, target_layer=1, dilation_px=3)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    # Some pixels near the boundary should be marked.
    assert (mask > 0).any()


def test_inpaint_mask_layer0_is_empty():
    layers = np.zeros((10, 10), dtype=np.int8)
    mask = compute_layer_inpaint_mask(layers, target_layer=0)
    assert (mask == 0).all()


def test_opencv_inpaint_fills_mask():
    """OpenCV inpainter should produce non-zero pixels where the mask is."""
    H, W = 50, 50
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    # Black hole in the center.
    image[20:30, 20:30] = 0
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[20:30, 20:30] = 255

    result = inpaint_opencv(image, mask, radius=5)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    # The inpainted region should no longer be all-zero.
    assert result[25, 25].sum() > 0


def test_inpaint_all_layers_opencv():
    """End-to-end: inpaint_all_layers should return one texture per layer."""
    H, W = 40, 40
    image = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    layers = np.zeros((H, W), dtype=np.int8)
    layers[:, :15] = 0
    layers[:, 15:30] = 1
    layers[:, 30:] = 2

    textures = inpaint_all_layers(
        image, layers, num_layers=3, backend="opencv", dilation_px=5,
    )
    assert set(textures.keys()) == {0, 1, 2}
    for k, tex in textures.items():
        assert tex.shape == image.shape
        assert tex.dtype == np.uint8
    # Layer 0 should be unchanged.
    np.testing.assert_array_equal(textures[0], image)
    # Layers 1 and 2 should differ from original (inpainted).
    assert not np.array_equal(textures[1], image)
