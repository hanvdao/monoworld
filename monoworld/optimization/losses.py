"""Loss functions for 3DGS optimization.

The total loss is:
    L = λ1 · L1(render, target)
      + λs · (1 - SSIM(render, target))
      + λp · LPIPS(render, target)
      + λd · L_depth  (ablation only; follows Chung et al. 2024)

The 3DGS paper uses λ1=0.8, λs=0.2, λp=0, λd=0.
H3 ablation toggles λd > 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# SSIM — standard 11x11 Gaussian window, stable implementation.
# ----------------------------------------------------------------------

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)  # (W, W)


_SSIM_KERNEL_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity. Inputs are (H, W, 3) or (3, H, W) in [0, 1].

    Returns a scalar mean SSIM.
    """
    if pred.shape[-1] == 3 and pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    key = (pred.device.type, pred.dtype)
    if key not in _SSIM_KERNEL_CACHE:
        _SSIM_KERNEL_CACHE[key] = (
            _gaussian_kernel(window_size).to(pred.device).to(pred.dtype)
        )
    kernel = _SSIM_KERNEL_CACHE[key]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.conv2d(pred, kernel, padding=window_size // 2, groups=pred.shape[1])
    mu_y = F.conv2d(target, kernel, padding=window_size // 2, groups=pred.shape[1])
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = (
        F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu_x2
    )
    sigma_y2 = (
        F.conv2d(target * target, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu_y2
    )
    sigma_xy = (
        F.conv2d(pred * target, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu_xy
    )
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean()


# ----------------------------------------------------------------------
# LPIPS — optional, requires `pip install lpips`.
# ----------------------------------------------------------------------

_LPIPS_NET = None


def _get_lpips(device: str):
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import lpips  # lazy import

        _LPIPS_NET = lpips.LPIPS(net="vgg").to(device).eval()
        for p in _LPIPS_NET.parameters():
            p.requires_grad = False
    return _LPIPS_NET


def lpips_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """LPIPS (VGG). Inputs (H, W, 3) in [0, 1]. Returns a scalar.

    Falls back to zero-tensor if `lpips` is not installed (keeps training
    runnable without it).
    """
    try:
        net = _get_lpips(pred.device.type)
    except ImportError:
        return torch.tensor(0.0, device=pred.device)

    # lpips expects (B, 3, H, W) in [-1, 1].
    pred_nhwc_to_nchw = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    target_nhwc_to_nchw = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return net(pred_nhwc_to_nchw, target_nhwc_to_nchw).mean()


# ----------------------------------------------------------------------
# Top-level loss
# ----------------------------------------------------------------------

def compute_photometric_loss(
    rendered: torch.Tensor,   # (H, W, 3) in [0, 1]
    target: torch.Tensor,     # (H, W, 3) in [0, 1]
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
    lambda_lpips: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Composite photometric loss matching 3DGS paper + optional LPIPS.

    Returns a dict with each component + 'total' for logging.
    """
    l1 = F.l1_loss(rendered, target)
    ssim_val = ssim(rendered, target)
    ssim_loss = 1 - ssim_val

    total = lambda_l1 * l1 + lambda_ssim * ssim_loss
    out = {"l1": l1, "ssim": ssim_val, "ssim_loss": ssim_loss, "total": total}

    if lambda_lpips > 0:
        lp = lpips_loss(rendered, target)
        total = total + lambda_lpips * lp
        out["lpips"] = lp
        out["total"] = total

    return out


def compute_depth_loss(
    rendered_depth: torch.Tensor,    # (H, W)
    target_depth: torch.Tensor,      # (H, W)
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depth regularization (Chung et al. 2024).

    Scale-and-shift aligned L2 on depth. This is the H3 ablation term.
    """
    if valid_mask is not None:
        rd = rendered_depth[valid_mask]
        td = target_depth[valid_mask]
    else:
        rd = rendered_depth.flatten()
        td = target_depth.flatten()

    # Scale+shift alignment (otherwise arbitrary MDE scale kills the loss).
    # Solve rd ≈ a * td + b in closed form.
    t_mean = td.mean()
    r_mean = rd.mean()
    num = ((td - t_mean) * (rd - r_mean)).sum()
    den = ((td - t_mean) ** 2).sum() + 1e-8
    a = num / den
    b = r_mean - a * t_mean
    aligned = a * td + b
    return ((rd - aligned) ** 2).mean()


# ----------------------------------------------------------------------
# Metrics (for logging, no gradients)
# ----------------------------------------------------------------------

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. Inputs in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 99.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()
