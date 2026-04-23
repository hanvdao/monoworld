"""Differentiable rasterizer wrapper around gsplat.

This module adapts our 3DGS `.ply` format (produced by Phase 0-6's
monoworld.scene.gaussian_splatting) to the tensor layout that gsplat
expects, and provides the inverse save path so the optimizer's output
is viewable in our existing browser.

gsplat requires CUDA. On Mac/CPU, optimization falls back to a simple
PyTorch-autograd point-splat renderer (SimpleRasterizer) — much slower
but keeps the code runnable on a laptop for debugging.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


class GaussianParams(NamedTuple):
    """Trainable 3DGS parameter tensors. All are nn.Parameters.

    Shapes:
        means:    (N, 3)    world-space centers
        scales:   (N, 3)    log-scales (stored as log for unconstrained opt)
        quats:    (N, 4)    unit quaternions (wxyz)
        opacities:(N, 1)    opacity logits (sigmoid -> [0,1])
        sh_dc:    (N, 1, 3) spherical harmonic DC band
        sh_rest:  (N, 0, 3) we use only DC (no view-dependent color)
    """
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor


# ----------------------------------------------------------------------
# Loaders / savers
# ----------------------------------------------------------------------

def load_splat_ply(path: str | Path, device: str = "cuda") -> GaussianParams:
    """Parse a 3DGS-format .ply from our init pipeline into trainable tensors.

    This matches the header written by monoworld.scene.gaussian_splatting
    (17 floats per vertex: xyz, nxnynz, f_dc_0..2, opacity, scale_0..2,
    rot_0..3). Byte-for-byte compatible with the viewer.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    # Find end of header.
    idx = raw.find(b"end_header\n")
    if idx < 0:
        raise ValueError(f"No end_header in {path}")
    header = raw[: idx].decode("ascii")
    body = raw[idx + len(b"end_header\n") :]

    # Parse vertex count.
    n = None
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
            break
    if n is None:
        raise ValueError(f"No vertex count in {path}")

    # 17 floats per vertex, little-endian.
    data = np.frombuffer(body, dtype="<f4").reshape(n, 17)

    xyz = data[:, 0:3]
    # 3:6 are normals — unused by 3DGS, discard.
    f_dc = data[:, 6:9]
    opacity = data[:, 9:10]
    scales = data[:, 10:13]
    rot = data[:, 13:17]

    return _to_params(xyz, f_dc, opacity, scales, rot, device=device)


def random_init_gaussians(
    n_points: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    device: str = "cuda",
    seed: int = 42,
) -> GaussianParams:
    """3DGS-paper-style random init: uniform points in bbox, random colors.

    Scale initialized to 1% of the scene's longest axis, matching the
    standard baseline from Kerbl et al. 2023.
    """
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(bounds_min, bounds_max, size=(n_points, 3)).astype(np.float32)

    # Random colors in [0,1], convert to SH DC.
    rgb = rng.uniform(0, 1, size=(n_points, 3)).astype(np.float32)
    f_dc = (rgb - 0.5) / SH_C0

    # Opacity 0.1 (logit ~= -2.2). The 3DGS paper starts low and lets
    # optimization raise it.
    opacity = np.full((n_points, 1), -2.2, dtype=np.float32)

    # Scale: 1% of bbox diagonal.
    diag = float(np.linalg.norm(bounds_max - bounds_min))
    scale_init = 0.01 * diag
    log_scales = np.full((n_points, 3), np.log(scale_init), dtype=np.float32)

    # Identity quaternion (wxyz).
    rot = np.zeros((n_points, 4), dtype=np.float32)
    rot[:, 0] = 1.0

    return _to_params(xyz, f_dc, opacity, log_scales, rot, device=device)


def _to_params(
    xyz: np.ndarray,
    f_dc: np.ndarray,
    opacity: np.ndarray,
    log_scales: np.ndarray,
    rot: np.ndarray,
    device: str,
) -> GaussianParams:
    """Wrap numpy arrays as nn.Parameters on the target device."""
    n = xyz.shape[0]
    means = nn.Parameter(torch.from_numpy(xyz).to(device))
    scales = nn.Parameter(torch.from_numpy(log_scales).to(device))
    quats = nn.Parameter(torch.from_numpy(rot).to(device))
    opacities = nn.Parameter(torch.from_numpy(opacity).to(device))
    sh_dc = nn.Parameter(
        torch.from_numpy(f_dc).view(n, 1, 3).to(device)
    )
    # Degree-0 only (no view dependence) — keeps the optimization tractable
    # with a single training view. Empty tensor for the rest bands.
    sh_rest = nn.Parameter(torch.zeros(n, 0, 3, device=device))
    return GaussianParams(means, scales, quats, opacities, sh_dc, sh_rest)


def save_splat_ply(params: GaussianParams, path: str | Path) -> None:
    """Write the optimized Gaussians back to our .ply format.

    Byte-compatible with monoworld.scene.gaussian_splatting.generate_splat_ply
    so the viewer loads it without changes.
    """
    path = Path(path)
    n = params.means.shape[0]
    xyz = params.means.detach().cpu().numpy()
    f_dc = params.sh_dc.detach().cpu().numpy().reshape(n, 3)
    opacity = params.opacities.detach().cpu().numpy()
    # Our viewer expects isotropic scale (uses avg of xyz). We stored
    # log-scales — write average.
    log_scales = params.scales.detach().cpu().numpy()
    rot = params.quats.detach().cpu().numpy()

    normals = np.zeros((n, 3), dtype=np.float32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack(
                "<17f",
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                f_dc[i, 0], f_dc[i, 1], f_dc[i, 2],
                float(opacity[i, 0]),
                log_scales[i, 0], log_scales[i, 1], log_scales[i, 2],
                rot[i, 0], rot[i, 1], rot[i, 2], rot[i, 3],
            ))


# ----------------------------------------------------------------------
# Rasterization — gsplat if available, else PyTorch fallback
# ----------------------------------------------------------------------

def render(
    params: GaussianParams,
    intrinsics: np.ndarray,  # (3, 3)
    width: int,
    height: int,
    c2w: torch.Tensor | None = None,  # (4, 4), default identity
    return_depth: bool = False,
) -> dict[str, torch.Tensor]:
    """Rasterize the Gaussians to an image.

    Returns:
        {"rgb": (H, W, 3) in [0,1], "depth": (H, W) if return_depth,
         "alpha": (H, W), "meta": dict with radii/visibility for densification}
    """
    try:
        from gsplat import rasterization
    except ImportError:
        return _render_pytorch_fallback(
            params, intrinsics, width, height, c2w, return_depth
        )
    return _render_gsplat(
        params, intrinsics, width, height, c2w, return_depth, rasterization
    )


def _render_gsplat(
    params: GaussianParams,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    c2w: torch.Tensor | None,
    return_depth: bool,
    rasterization,  # gsplat.rasterization
) -> dict[str, torch.Tensor]:
    """Render via the real gsplat rasterizer (CUDA)."""
    device = params.means.device
    if c2w is None:
        c2w = torch.eye(4, device=device, dtype=torch.float32)

    # gsplat expects view matrix (world-to-camera) and K batched.
    viewmat = torch.inverse(c2w).unsqueeze(0)  # (1, 4, 4)
    K = torch.from_numpy(intrinsics).float().unsqueeze(0).to(device)  # (1, 3, 3)

    # Compose SH coefficients (DC only for our single-view setting).
    sh = params.sh_dc  # (N, 1, 3). gsplat accepts this as sh_degree=0.

    # Normalize opacity to [0,1] via sigmoid, scales via exp.
    opacities = torch.sigmoid(params.opacities).squeeze(-1)  # (N,)
    scales = torch.exp(params.scales)  # (N, 3)
    quats = torch.nn.functional.normalize(params.quats, dim=-1)

    render_mode = "RGB+ED" if return_depth else "RGB"
    renders, alphas, meta = rasterization(
        means=params.means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=0,
        render_mode=render_mode,
        packed=False,
    )
    # renders: (1, H, W, 3) or (1, H, W, 4) with depth.
    renders = renders.squeeze(0)
    alphas = alphas.squeeze(0).squeeze(-1)
    out = {"rgb": renders[..., :3].clamp(0, 1), "alpha": alphas, "meta": meta}
    if return_depth:
        out["depth"] = renders[..., 3]
    return out


def _render_pytorch_fallback(
    params: GaussianParams,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    c2w: torch.Tensor | None,
    return_depth: bool,
) -> dict[str, torch.Tensor]:
    """Simple point-splat renderer using pure PyTorch autograd.

    No per-Gaussian anisotropy — treats each Gaussian as an isotropic
    alpha-blended disk. Much slower and lower quality than gsplat but
    lets the code run on MPS/CPU for local debugging.

    This is a *fallback*, not a contribution. Do all real experiments
    on a CUDA GPU with real gsplat.
    """
    device = params.means.device
    if c2w is None:
        c2w = torch.eye(4, device=device, dtype=torch.float32)

    # World -> camera.
    w2c = torch.inverse(c2w)
    xyz = params.means  # (N, 3)
    ones = torch.ones(xyz.shape[0], 1, device=device)
    xyz_h = torch.cat([xyz, ones], dim=-1)  # (N, 4)
    xyz_cam = (w2c @ xyz_h.T).T[:, :3]  # (N, 3)

    # Behind camera.
    z = xyz_cam[:, 2]
    visible = z > 0.05

    # Project to pixels.
    K = torch.from_numpy(intrinsics).float().to(device)
    uvs = (K @ xyz_cam.T).T  # (N, 3)
    u = uvs[:, 0] / uvs[:, 2]
    v = uvs[:, 1] / uvs[:, 2]

    # Screen-space scale ~ world scale / depth * focal.
    scales_w = torch.exp(params.scales).mean(dim=-1)  # (N,)
    fx = float(intrinsics[0, 0])
    radius_px = (scales_w * fx / z.clamp(min=0.05)).clamp(1.0, 64.0)

    # Accumulate into an image. We do this with scatter-add over soft disks.
    H, W = height, width
    img = torch.zeros(H, W, 3, device=device)
    alpha_accum = torch.zeros(H, W, device=device)

    colors = params.sh_dc.squeeze(1) * SH_C0 + 0.5  # (N, 3) in [0, 1]
    colors = colors.clamp(0, 1)
    opacities = torch.sigmoid(params.opacities).squeeze(-1)  # (N,)

    # Sort back-to-front by depth.
    order = torch.argsort(z, descending=True)
    for idx in order[visible[order]]:
        cx, cy = int(u[idx].item()), int(v[idx].item())
        r = int(radius_px[idx].item())
        if cx < -r or cx >= W + r or cy < -r or cy >= H + r:
            continue
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys = torch.arange(y0, y1, device=device).float() - cy
        xs = torch.arange(x0, x1, device=device).float() - cx
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        dist2 = (gx ** 2 + gy ** 2) / (radius_px[idx] ** 2 + 1e-6)
        weight = torch.exp(-4.0 * dist2) * opacities[idx]
        w3 = weight.unsqueeze(-1) * colors[idx].unsqueeze(0).unsqueeze(0)
        img[y0:y1, x0:x1] += w3 * (1.0 - alpha_accum[y0:y1, x0:x1].unsqueeze(-1))
        alpha_accum[y0:y1, x0:x1] += weight * (1.0 - alpha_accum[y0:y1, x0:x1])

    out = {"rgb": img.clamp(0, 1), "alpha": alpha_accum, "meta": {}}
    if return_depth:
        out["depth"] = torch.zeros(H, W, device=device)  # not computed in fallback
    return out
