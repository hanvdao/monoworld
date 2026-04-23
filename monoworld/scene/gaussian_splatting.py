"""Gaussian Splatting scene generation.

Converts a colored 3D point cloud into a Gaussian Splat representation stored
as a .ply file in the standard 3D Gaussian Splatting format. Each point
becomes an isotropic (spherical) Gaussian with:
    - position: from the depth-unprojected point cloud
    - color: from the source image (stored as spherical harmonics DC)
    - scale: estimated from local point spacing
    - rotation: identity quaternion
    - opacity: high (0.9)

No optimization is performed — this is a "direct initialization" that already
looks dramatically better than a triangle mesh because:
    1. Gaussians are soft/fuzzy, so depth discontinuities blend instead of tearing
    2. Alpha compositing naturally handles transparency at silhouette edges
    3. The per-point representation avoids the topological artifacts of meshing

The output .ply follows the format established by the original 3DGS paper
(Kerbl et al., 2023) and is loadable by all standard Gaussian Splat viewers.

For even better quality, the initialized Gaussians can be optimized with a
differentiable renderer (e.g. gsplat, nerfstudio), but the unoptimized
version is already a significant improvement over meshes.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


# Spherical harmonics coefficient for DC band (degree 0).
SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def rgb_to_sh_dc(rgb_float: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to spherical harmonics DC coefficients.

    The 3DGS paper stores color as SH coefficients. For DC-only (no
    view-dependent effects), the conversion is:
        sh_dc = (rgb - 0.5) / SH_C0
    """
    return (rgb_float - 0.5) / SH_C0


def estimate_point_scale(
    points: np.ndarray,
    default_scale: float = 0.01,
    k_neighbors: int = 3,
    max_points_for_knn: int = 50000,
) -> np.ndarray:
    """Estimate per-point scale from local point density.

    For each point, finds the k nearest neighbors and uses the mean distance
    as the Gaussian's scale. For large point clouds, subsamples for speed.

    Returns:
        (N,) float32 array of per-point scales.
    """
    N = len(points)
    if N < k_neighbors + 1:
        return np.full(N, default_scale, dtype=np.float32)

    # For very large clouds, estimate globally from a random sample.
    if N > max_points_for_knn:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, max_points_for_knn, replace=False)
        sample = points[idx]
        # Compute pairwise distances on sample to get median spacing.
        from scipy.spatial import cKDTree
        tree = cKDTree(sample)
        dists, _ = tree.query(sample, k=k_neighbors + 1)
        median_dist = float(np.median(dists[:, 1:]))  # skip self
        return np.full(N, median_dist * 0.7, dtype=np.float32)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k_neighbors + 1)
        # Mean distance to k nearest (skip index 0 which is self).
        scales = np.mean(dists[:, 1:], axis=1).astype(np.float32) * 0.7
        return scales
    except ImportError:
        # scipy not available — use a global estimate.
        return np.full(N, default_scale, dtype=np.float32)


def generate_splat_ply(
    points: np.ndarray,
    colors: np.ndarray,
    out_path: str | Path,
    scale_override: float | None = None,
    opacity: float = 0.9,
) -> int:
    """Generate a 3DGS-compatible .ply from a colored point cloud.

    Args:
        points: (N, 3) float32 positions.
        colors: (N, 3) float32 RGB in [0, 1].
        out_path: output .ply path.
        scale_override: if set, use this fixed scale for all Gaussians
            instead of estimating from point density.
        opacity: initial opacity for all Gaussians (0-1).

    Returns:
        Number of Gaussians written.
    """
    out_path = Path(out_path)
    N = len(points)
    if N == 0:
        raise ValueError("Empty point cloud.")
    if points.shape != (N, 3) or colors.shape != (N, 3):
        raise ValueError(f"Shape mismatch: points={points.shape}, colors={colors.shape}")

    # Compute scales.
    if scale_override is not None:
        scales = np.full(N, scale_override, dtype=np.float32)
    else:
        scales = estimate_point_scale(points)

    # Convert color to SH DC.
    sh_dc = rgb_to_sh_dc(colors.astype(np.float32))  # (N, 3)

    # Opacity as logit: logit(p) = log(p / (1 - p)).
    opacity_logit = np.log(opacity / (1.0 - opacity + 1e-8))

    # Log scale (3DGS stores log of scale).
    log_scales = np.log(np.clip(scales, 1e-7, None))

    # Identity quaternion (w, x, y, z) = (1, 0, 0, 0) for all.
    rot = np.zeros((N, 4), dtype=np.float32)
    rot[:, 0] = 1.0  # w = 1

    # Normals (unused but required by format).
    normals = np.zeros((N, 3), dtype=np.float32)

    # Write PLY.
    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            # Pack each Gaussian as a sequence of floats.
            f.write(struct.pack(
                "<17f",
                points[i, 0], points[i, 1], points[i, 2],       # position
                normals[i, 0], normals[i, 1], normals[i, 2],     # normals
                sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2],          # color (SH DC)
                opacity_logit,                                    # opacity
                log_scales[i], log_scales[i], log_scales[i],     # scale (isotropic)
                rot[i, 0], rot[i, 1], rot[i, 2], rot[i, 3],     # rotation
            ))

    return N


def pointcloud_to_splat(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics,
    out_path: str | Path,
    stride: int = 2,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
    opacity: float = 0.9,
) -> int:
    """End-to-end: image + depth -> Gaussian Splat .ply.

    Convenience function that backprojects the depth map, colors the points,
    and writes the splat file. Uses the same unprojection code as the mesh
    pipeline but outputs Gaussians instead of triangles.

    Args:
        image_rgb: HxWx3 uint8 source image.
        depth: HxW float32 depth map.
        intrinsics: Intrinsics object.
        out_path: output .ply path.
        stride: pixel stride (controls point count). 2 = quarter resolution.
        min_depth, max_depth: depth range filter.
        opacity: per-Gaussian opacity.

    Returns:
        Number of Gaussians written.
    """
    from ..geometry.unproject import unproject_depth_to_points

    H, W = depth.shape
    points, valid_flat = unproject_depth_to_points(
        depth, intrinsics, stride=stride,
        min_depth=min_depth, max_depth=max_depth,
    )
    colors = image_rgb[::stride, ::stride].reshape(-1, 3).astype(np.float32) / 255.0
    colors = colors[valid_flat]

    return generate_splat_ply(points, colors, out_path, opacity=opacity)
