"""Segmentation + scene layering."""
from .layering import (
    assign_depth_layers,
    colorize_layer_mask,
    layer_color_legend,
)

__all__ = [
    "assign_depth_layers",
    "colorize_layer_mask",
    "layer_color_legend",
]
