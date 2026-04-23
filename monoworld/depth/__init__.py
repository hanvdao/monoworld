from .midas import MiDaSDepthEstimator
from .utils import (
    disparity_to_depth,
    normalize_depth_for_vis,
    colorize_depth,
    resolve_device,
)

# DepthAnythingV2 is optional — only available if `transformers` is installed.
try:
    from .depth_anything import DepthAnythingV2Estimator
except ImportError:
    DepthAnythingV2Estimator = None  # type: ignore[misc,assignment]

__all__ = [
    "MiDaSDepthEstimator",
    "DepthAnythingV2Estimator",
    "disparity_to_depth",
    "normalize_depth_for_vis",
    "colorize_depth",
    "resolve_device",
]
