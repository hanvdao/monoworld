"""Depth Anything V2 depth estimator (upgrade from MiDaS).

Uses the HuggingFace transformers pipeline. Produces significantly sharper
depth boundaries than MiDaS, which directly reduces silhouette artifacts
in the 3D reconstruction.

Install: pip install transformers
Model downloads ~1.3 GB on first run.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .utils import disparity_to_depth, resolve_device


class DepthAnythingV2Estimator:
    """Depth Anything V2 monocular depth estimator.

    Example:
        est = DepthAnythingV2Estimator()
        depth = est.infer(rgb_uint8)   # HxW float32 depth in [near, far]
    """

    # Available sizes: Small, Base, Large. Large is best quality.
    MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"

    def __init__(
        self,
        device: str = "auto",
        near: float = 0.5,
        far: float = 20.0,
        model_id: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.near = near
        self.far = far
        self.model_id = model_id or self.MODEL_ID

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError(
                "Depth Anything V2 requires 'transformers'. "
                "Install with: pip install transformers"
            )

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run depth inference on an HxWx3 uint8 RGB image.

        Returns:
            HxW float32 depth map in [near, far].
        """
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3:
            raise ValueError("Expected HxWx3 uint8 RGB image.")
        H, W = image_rgb.shape[:2]

        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Resize to original image size.
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        disparity = prediction.cpu().numpy().astype(np.float32)
        depth = disparity_to_depth(disparity, near=self.near, far=self.far)
        return depth
