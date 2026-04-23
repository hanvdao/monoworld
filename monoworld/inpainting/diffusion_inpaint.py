"""Stable Diffusion inpainting backend (optional, high quality).

Uses runwayml/stable-diffusion-inpainting (SD 1.5 Inpaint) via diffusers.
Requires ~4 GB VRAM (float16) or ~8 GB RAM (float32 CPU). On Apple Silicon
Macs, MPS backend is used automatically.

This module gracefully degrades: if diffusers is not installed, importing
still works but calling inpaint_diffusion() raises a clear error.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Lazy-import diffusers so the rest of the project works without it.
_PIPELINE = None
_PIPELINE_DEVICE = None


def _ensure_pipeline(device: str = "auto") -> tuple:
    """Lazy-load the SD Inpaint pipeline. Cached across calls."""
    global _PIPELINE, _PIPELINE_DEVICE

    if _PIPELINE is not None and _PIPELINE_DEVICE == device:
        return _PIPELINE, _PIPELINE_DEVICE

    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError:
        raise ImportError(
            "Diffusion inpainting requires 'diffusers' and 'transformers'. "
            "Install with: pip install diffusers transformers accelerate"
        )

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,        # skip NSFW filter for speed
        requires_safety_checker=False,
    )
    pipe.to(device)

    # Memory optimizations.
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not installed, that's fine

    _PIPELINE = pipe
    _PIPELINE_DEVICE = device
    return pipe, device


def inpaint_diffusion(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    prompt: str = "photorealistic background, high quality, detailed",
    negative_prompt: str = "blurry, artifacts, distortion, text, watermark",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    device: str = "auto",
    seed: int | None = 42,
) -> np.ndarray:
    """Inpaint masked regions using Stable Diffusion.

    Args:
        image_rgb: HxWx3 uint8 RGB image.
        mask: HxW uint8 mask (255 = inpaint here, 0 = keep original).
        prompt: text prompt to guide generation.
        negative_prompt: what to avoid.
        num_inference_steps: diffusion steps (more = higher quality, slower).
        guidance_scale: classifier-free guidance weight.
        device: "auto", "cuda", "mps", or "cpu".
        seed: random seed for reproducibility. None = random.

    Returns:
        HxWx3 uint8 RGB inpainted image.
    """
    import torch

    pipe, device = _ensure_pipeline(device)

    H, W = image_rgb.shape[:2]
    # SD requires dimensions divisible by 8.
    new_h = (H // 8) * 8
    new_w = (W // 8) * 8

    pil_image = Image.fromarray(image_rgb).resize((new_w, new_h), Image.LANCZOS)
    pil_mask = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)

    generator = torch.Generator(device=device if device != "mps" else "cpu")
    if seed is not None:
        generator = generator.manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        mask_image=pil_mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=new_h,
        width=new_w,
    ).images[0]

    # Resize back to original dimensions.
    result = result.resize((W, H), Image.LANCZOS)
    return np.array(result, dtype=np.uint8)
