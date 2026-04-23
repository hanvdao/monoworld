"""Build data/outputs/scenes.json — an index the viewer fetches at startup.

Run automatically by run.py after each pipeline invocation. Can also be
invoked manually:

    python scripts/build_scene_index.py
    python scripts/build_scene_index.py --root data/outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_index(root: Path) -> dict:
    """Scan `root` for scene subdirectories containing pointcloud.ply.

    Each scene becomes an entry with paths relative to `root` so the viewer
    can fetch them at /<scene_id>/pointcloud.ply (because Vite serves
    `data/outputs/` as its publicDir).
    """
    scenes = []
    if not root.exists():
        return {"scenes": []}

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        ply = scene_dir / "pointcloud.ply"
        glb = scene_dir / "mesh.glb"
        splat = scene_dir / "splat.ply"
        meta = scene_dir / "metadata.json"
        if not ply.exists() and not glb.exists() and not splat.exists():
            continue
        scenes.append({
            "id": scene_dir.name,
            "ply": f"/{scene_dir.name}/pointcloud.ply" if ply.exists() else None,
            "glb": f"/{scene_dir.name}/mesh.glb" if glb.exists() else None,
            "splat": f"/{scene_dir.name}/splat.ply" if splat.exists() else None,
            "metadata": f"/{scene_dir.name}/metadata.json" if meta.exists() else None,
        })
    return {"scenes": scenes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenes.json index.")
    parser.add_argument("--root", default="data/outputs", type=str)
    args = parser.parse_args()

    root = Path(args.root)
    index = build_index(root)
    out_path = root / "scenes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {out_path} ({len(index['scenes'])} scene(s))")


if __name__ == "__main__":
    main()
