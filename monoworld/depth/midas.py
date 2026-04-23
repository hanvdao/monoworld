"""MiDaS depth estimator (Phase 1 baseline).

Uses intel-isl/MiDaS via torch.hub. Returns disparity-like relative depth,
which we convert to metric-ish depth via disparity_to_depth().
"""
from __future__ import annotations

import numpy as np
import torch

from .utils import disparity_to_depth, resolve_device


class MiDaSDepthEstimator:
    """Pretrained monocular depth estimator.

    Example:
        est = MiDaSDepthEstimator()
        depth = est.infer(rgb_uint8)   # HxW float32 depth in [near, far]
    """

    HUB_REPO = "intel-isl/MiDaS"
    MODEL_NAME = "DPT_Large"   # alt: "DPT_Hybrid" (lighter), "MiDaS_small" (tiny)

    def __init__(
        self,
        device: str = "auto",
        near: float = 0.5,
        far: float = 20.0,
        model_name: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.near = near
        self.far = far
        self.model_name = model_name or self.MODEL_NAME

        # Load model + matching transforms.
        self.model = torch.hub.load(self.HUB_REPO, self.model_name)
        self.model.to(self.device).eval()

        transforms = torch.hub.load(self.HUB_REPO, "transforms")
        if self.model_name.startswith("DPT"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    @torch.inference_mode()
    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run depth inference on an HxWx3 uint8 RGB image.

        Returns:
            HxW float32 depth map in [near, far].
        """
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3:
            raise ValueError("Expected HxWx3 uint8 RGB image.")
        H, W = image_rgb.shape[:2]

        batch = self.transform(image_rgb).to(self.device)
        prediction = self.model(batch)  # (1, h, w) disparity

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        disparity = prediction.detach().cpu().numpy().astype(np.float32)
        depth = disparity_to_depth(disparity, near=self.near, far=self.far)
        return depth
