"""Diagnose why depth-init splats render poorly.

Renders the splat.ply at iteration 0 and saves:
  - target.png (source image)
  - render_init.png (what the splats look like BEFORE any training)
  - diff.png (pixel difference)

Also reports: actual scale range, opacity range, color range, camera.
Helps pinpoint whether the problem is geometry (Y-flip), scales, or opacity.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from monoworld.optimization.rasterizer import load_splat_ply, render
from monoworld.utils import load_image


def main(scene_dir: str):
    scene = Path(scene_dir)
    meta = json.loads((scene / "metadata.json").read_text())

    target = load_image(Path(meta["input_image"]))
    H, W = target.shape[:2]
    K = meta["intrinsics"]
    intrinsics = np.array(
        [[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]], dtype=np.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params = load_splat_ply(scene / "splat.ply", device=device)
    n = params.means.shape[0]

    # --- Print diagnostics ---
    means = params.means.detach().cpu().numpy()
    log_scales = params.scales.detach().cpu().numpy()
    actual_scales = np.exp(log_scales)
    opac_logits = params.opacities.detach().cpu().numpy()
    opac_sigmoid = 1 / (1 + np.exp(-opac_logits))
    sh_dc = params.sh_dc.detach().cpu().numpy().reshape(n, 3)
    SH_C0 = 0.28209479177387814
    colors = sh_dc * SH_C0 + 0.5

    print(f"\n=== DIAGNOSTICS for {n} Gaussians ===")
    print(f"Position range:")
    print(f"  X: [{means[:,0].min():7.2f} .. {means[:,0].max():7.2f}]  "
          f"mean={means[:,0].mean():+.2f}")
    print(f"  Y: [{means[:,1].min():7.2f} .. {means[:,1].max():7.2f}]  "
          f"mean={means[:,1].mean():+.2f}")
    print(f"  Z: [{means[:,2].min():7.2f} .. {means[:,2].max():7.2f}]  "
          f"mean={means[:,2].mean():+.2f}")
    print(f"  (for gsplat, Z > 0 means in front of camera)")
    print()
    print(f"Log-scale: [{log_scales.min():.3f} .. {log_scales.max():.3f}]")
    print(f"Actual scale: [{actual_scales.min():.5f} .. {actual_scales.max():.5f}]  "
          f"median={np.median(actual_scales):.5f}")
    print()
    print(f"Opacity logit: [{opac_logits.min():.3f} .. {opac_logits.max():.3f}]")
    print(f"Opacity sigmoid: [{opac_sigmoid.min():.3f} .. {opac_sigmoid.max():.3f}]")
    print()
    print(f"Color (SH_DC decoded to RGB): "
          f"[{colors.min():.3f} .. {colors.max():.3f}]  "
          f"mean=({colors[:,0].mean():.2f},{colors[:,1].mean():.2f},{colors[:,2].mean():.2f})")
    print()
    print(f"Intrinsics: fx={K['fx']:.1f} fy={K['fy']:.1f} "
          f"cx={K['cx']:.1f} cy={K['cy']:.1f}")

    # --- Render at identity camera ---
    with torch.no_grad():
        out = render(params, intrinsics, W, H)
    rendered = (out["rgb"].cpu().numpy() * 255).astype(np.uint8)
    alpha = (out["alpha"].cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(rendered).save(scene / "diag_render_init.png")
    Image.fromarray(target).save(scene / "diag_target.png")
    Image.fromarray(alpha).save(scene / "diag_alpha.png")

    # Per-channel MSE and PSNR.
    diff = rendered.astype(np.float32) - target.astype(np.float32)
    mse = (diff ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))
    print(f"\n=== INITIAL RENDER ===")
    print(f"Rendered range: [{rendered.min()} .. {rendered.max()}], "
          f"mean={rendered.mean():.1f}")
    print(f"Alpha coverage: {(alpha > 10).mean() * 100:.1f}% of pixels have alpha>10/255")
    print(f"Reprojection PSNR: {psnr:.2f} dB")
    print()
    print(f"Saved: {scene/'diag_render_init.png'}")
    print(f"Saved: {scene/'diag_target.png'}")
    print(f"Saved: {scene/'diag_alpha.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/outputs/test_*")
