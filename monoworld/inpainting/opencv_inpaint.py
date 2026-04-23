"""OpenCV-based inpainting (fast fallback).

Uses the Telea or Navier-Stokes algorithm built into OpenCV. Quality is
decent for small holes (< ~30px) but produces blurry results for large
disocclusion regions. Good enough for a demo, and runs instantly.
"""
from __future__ import annotations

import cv2
import numpy as np


def inpaint_opencv(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
    method: str = "telea",
) -> np.ndarray:
    """Inpaint masked regions of an RGB image using OpenCV.

    Args:
        image_rgb: HxWx3 uint8 RGB image.
        mask: HxW uint8 mask (255 = inpaint here, 0 = keep original).
        radius: inpainting neighborhood radius.
        method: "telea" or "ns" (Navier-Stokes).

    Returns:
        HxWx3 uint8 RGB inpainted image.
    """
    # OpenCV works in BGR.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    result_bgr = cv2.inpaint(image_bgr, mask, radius, flag)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
