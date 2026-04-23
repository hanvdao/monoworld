"""Per-layer inpainting pipeline.

Takes the source image and layer assignments, computes disocclusion masks for
each background layer, inpaints them, and returns per-layer texture images.

The foreground layer (layer 0) keeps its original pixels — there's nothing
behind the camera to reveal. Background layers (1, 2, ...) get inpainted
at locations where foreground objects occlude them.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.logging import get_logger
from .disocclusion import compute_layer_inpaint_mask
from .opencv_inpaint import inpaint_opencv

log = get_logger("monoworld.inpainting")


def inpaint_all_layers(
    image_rgb: np.ndarray,
    layer_mask: np.ndarray,
    num_layers: int,
    backend: str = "opencv",
    dilation_px: int = 15,
    prompt: str = "photorealistic background, high quality, detailed",
    device: str = "auto",
) -> dict[int, np.ndarray]:
    """Produce an inpainted texture image for each layer.

    Args:
        image_rgb: HxWx3 uint8 source image.
        layer_mask: HxW int8 layer assignment.
        num_layers: total number of layers.
        backend: "opencv" (fast fallback) or "diffusion" (SD 1.5 Inpaint).
        dilation_px: pixels to dilate foreground mask for disocclusion.
        prompt: text prompt for diffusion backend (ignored by opencv).
        device: device for diffusion backend.

    Returns:
        {layer_id: HxWx3 uint8 RGB image} for each layer.
        Layer 0 returns the original image unchanged.
        Layers 1+ return the image with foreground regions inpainted.
    """
    textures: dict[int, np.ndarray] = {}

    for k in range(num_layers):
        if k == 0:
            # Foreground layer: nothing to inpaint, use original.
            textures[k] = image_rgb.copy()
            log.info("Layer %d: foreground — using original image", k)
            continue

        # Compute the inpaint mask for this background layer.
        mask = compute_layer_inpaint_mask(layer_mask, k, dilation_px=dilation_px)
        mask_pixels = int((mask > 0).sum())
        total_pixels = mask.size
        pct = 100.0 * mask_pixels / total_pixels
        log.info("Layer %d: inpaint mask covers %d px (%.1f%%)", k, mask_pixels, pct)

        if mask_pixels == 0:
            # Nothing to inpaint.
            textures[k] = image_rgb.copy()
            continue

        if backend == "opencv":
            result = inpaint_opencv(image_rgb, mask, radius=7)
        elif backend == "diffusion":
            from .diffusion_inpaint import inpaint_diffusion
            result = inpaint_diffusion(
                image_rgb, mask,
                prompt=prompt,
                device=device,
            )
        else:
            raise ValueError(f"Unknown inpaint backend: {backend}")

        textures[k] = result

    return textures
