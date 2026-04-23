"""Open an existing .ply in Open3D's viewer.

Usage:
    python scripts/preview_ply.py data/outputs/<scene_id>/pointcloud.ply
"""
from __future__ import annotations

import sys
from pathlib import Path

import open3d as o3d


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/preview_ply.py <path-to-ply>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(str(path))
    print(f"Loaded {len(pcd.points)} points from {path}")
    o3d.visualization.draw_geometries([pcd], window_name=str(path.name))


if __name__ == "__main__":
    main()
