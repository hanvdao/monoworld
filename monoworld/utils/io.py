"""I/O helpers: image loading, config parsing, output path management."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an RGB image as a uint8 HxWx3 numpy array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def scene_id_from_path(image_path: str | Path) -> str:
    """Deterministic scene id from the input image path + timestamp.

    Collision-safe across runs on the same image because of the timestamp,
    while staying short and filesystem-safe.
    """
    image_path = Path(image_path)
    stem = image_path.stem
    h = hashlib.md5(str(image_path.resolve()).encode()).hexdigest()[:6]
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{stem}_{h}_{ts}"
