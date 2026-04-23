"""Differentiable 3DGS optimization — the Path A systems contribution."""
from .losses import compute_photometric_loss, compute_depth_loss
from .train_loop import train
from .rasterizer import load_splat_ply, random_init_gaussians

__all__ = [
    "train",
    "compute_photometric_loss",
    "compute_depth_loss",
    "load_splat_ply",
    "random_init_gaussians",
]
