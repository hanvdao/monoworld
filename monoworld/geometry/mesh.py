"""Mesh construction from a depth map.

Pipeline:
    depth (HxW) + intrinsics
        -> per-pixel grid of 3D vertices
        -> two triangles per grid cell (cells with any invalid corner skipped)
        -> drop triangles whose longest edge >> median (kills stretch artifacts
           at depth discontinuities)
        -> UVs from normalized pixel coordinates so the source image can be
           used as a texture

Coordinate convention: vertices are returned in **Three.js convention**:
    +X right, +Y up, +Z out of screen (toward viewer).
This is achieved by negating Y after backprojection from OpenCV camera coords
(+Y down). The viewer can render the .glb directly with no flip.

We control triangle count by adjusting pixel stride at build time. This keeps
UVs trivially aligned to pixel centers and avoids the UV-corruption issues of
post-hoc quadric decimation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .intrinsics import Intrinsics


def auto_stride_for_budget(
    width: int,
    height: int,
    target_triangles: int = 200_000,
    min_stride: int = 1,
) -> int:
    """Choose pixel stride so the resulting mesh has ~target_triangles.

    Triangle count for an HxW depth grid at stride S:
        ~ 2 * (H/S - 1) * (W/S - 1)  ~  2 * H * W / S^2
    """
    s = math.sqrt(2.0 * width * height / max(target_triangles, 1))
    return max(min_stride, int(round(s)))


def build_triangle_mesh_from_depth(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
    edge_threshold_factor: float = 3.0,
    extra_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a triangle mesh from a depth map.

    Args:
        depth: HxW float depth map in [near, far].
        intrinsics: pinhole intrinsics matching the depth resolution.
        stride: pixel stride (1 = every pixel; higher = fewer triangles).
        min_depth, max_depth: depths outside this range mark invalid pixels.
        edge_threshold_factor: drop triangles whose longest edge exceeds
            `factor * median_longest_edge`. Set to 0 to disable.
        extra_valid_mask: optional HxW bool mask (True = include this pixel).
            Combined with depth-range validity. Used by per-layer meshing
            to restrict triangles to a single layer.

    Returns:
        vertices: (N, 3) float32 in **Three.js coords** (+Y up, +Z forward).
        triangles: (M, 3) int32 indices into vertices.
        uvs: (N, 2) float32 in [0, 1] (top-left origin, V down — glTF spec).
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth.shape}")

    H, W = depth.shape
    us = np.arange(0, W, stride, dtype=np.int32)
    vs = np.arange(0, H, stride, dtype=np.int32)
    h, w = len(vs), len(us)

    uu, vv = np.meshgrid(us.astype(np.float32), vs.astype(np.float32))
    z = depth[::stride, ::stride].astype(np.float32)

    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if extra_valid_mask is not None:
        valid = valid & extra_valid_mask[::stride, ::stride]

    # Backproject to camera coords (OpenCV: +X right, +Y down, +Z forward).
    X = (uu - intrinsics.cx) * z / intrinsics.fx
    Y = (vv - intrinsics.cy) * z / intrinsics.fy
    Z = z
    # Convert to Three.js coords by negating Y (+Y now points up).
    Y = -Y
    vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    # UVs: normalized pixel coords. glTF UV origin = top-left, V down,
    # which matches image pixel coords directly. No flip needed.
    uvs = np.stack([uu / max(W - 1, 1), vv / max(H - 1, 1)], axis=-1) \
            .reshape(-1, 2).astype(np.float32)

    # Triangle indexing per cell:
    #   tl --- tr
    #    | \    |
    #    |  \   |
    #   bl --- br
    i_idx, j_idx = np.meshgrid(
        np.arange(h - 1, dtype=np.int32),
        np.arange(w - 1, dtype=np.int32),
        indexing="ij",
    )
    i_idx = i_idx.reshape(-1)
    j_idx = j_idx.reshape(-1)

    tl = i_idx * w + j_idx
    tr = i_idx * w + (j_idx + 1)
    bl = (i_idx + 1) * w + j_idx
    br = (i_idx + 1) * w + (j_idx + 1)

    valid_flat = valid.reshape(-1)
    cell_valid = valid_flat[tl] & valid_flat[tr] & valid_flat[bl] & valid_flat[br]
    tl, tr, bl, br = tl[cell_valid], tr[cell_valid], bl[cell_valid], br[cell_valid]

    # Two triangles per cell. Note: because we negated Y, the original CCW
    # winding (in OpenCV coords) becomes CW (in Three.js coords). The viewer
    # handles this via DoubleSide, so winding doesn't matter.
    tri1 = np.stack([tl, tr, bl], axis=-1)
    tri2 = np.stack([tr, br, bl], axis=-1)
    triangles = np.concatenate([tri1, tri2], axis=0).astype(np.int32)

    # Edge-length culling: kills triangles spanning depth discontinuities.
    if edge_threshold_factor > 0 and len(triangles) > 0:
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        max_edge = np.maximum(np.maximum(e0, e1), e2)
        median_edge = float(np.median(max_edge))
        threshold = edge_threshold_factor * median_edge
        keep = max_edge < threshold
        triangles = triangles[keep]

    return vertices, triangles, uvs


def export_textured_glb(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    image_rgb: np.ndarray,
    out_path: str | Path,
) -> None:
    """Export a single textured triangle mesh as a binary glTF (.glb).

    The texture is the source RGB image, embedded inside the .glb.
    """
    out_path = Path(out_path)
    if vertices.shape[0] != uvs.shape[0]:
        raise ValueError(
            f"Vertex/UV count mismatch: {vertices.shape[0]} vs {uvs.shape[0]}"
        )

    texture = Image.fromarray(image_rgb)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=triangles,
        visual=visual,
        process=False,  # critical: process=True merges vertices and breaks UVs
    )
    mesh.export(out_path, file_type="glb")


def export_layered_glb(
    layers: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    image_rgb: np.ndarray,
    out_path: str | Path,
    per_layer_textures: dict[int, np.ndarray] | None = None,
) -> None:
    """Export a multi-mesh .glb where each layer is a separately-named mesh.

    Args:
        layers: dict of {layer_id: (vertices, triangles, uvs)}.
        image_rgb: HxWx3 uint8 source image (fallback texture for all layers).
        out_path: destination .glb path.
        per_layer_textures: optional {layer_id: HxWx3 uint8 RGB}. If provided,
            each layer uses its own inpainted texture instead of the source.

    The resulting glTF Scene contains one mesh per non-empty layer, named
    "layer_0", "layer_1", ... so the viewer can toggle them individually.
    """
    out_path = Path(out_path)

    scene = trimesh.Scene()
    n_meshes = 0
    for layer_id, (verts, tris, uvs) in sorted(layers.items()):
        if len(tris) == 0:
            continue
        tex_img = image_rgb
        if per_layer_textures and layer_id in per_layer_textures:
            tex_img = per_layer_textures[layer_id]
        texture = Image.fromarray(tex_img)
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture)
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=tris,
            visual=visual,
            process=False,
        )
        scene.add_geometry(mesh, geom_name=f"layer_{layer_id}",
                           node_name=f"layer_{layer_id}")
        n_meshes += 1

    if n_meshes == 0:
        raise ValueError("No non-empty layers to export.")

    scene.export(out_path, file_type="glb")


def build_per_layer_meshes(
    depth: np.ndarray,
    layer_mask: np.ndarray,
    intrinsics: Intrinsics,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
    edge_threshold_factor: float = 3.0,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build one triangle mesh per depth layer.

    Args:
        depth: HxW float depth map.
        layer_mask: HxW int8 layer assignment (-1 = invalid). From
            `monoworld.segmentation.assign_depth_layers`.
        intrinsics: matches depth resolution.
        stride, min/max_depth, edge_threshold_factor: passed through.

    Returns:
        {layer_id: (vertices, triangles, uvs)} for each layer that has
        triangles after culling.
    """
    layer_ids = sorted(int(k) for k in np.unique(layer_mask) if k >= 0)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in layer_ids:
        mask = layer_mask == k
        v, t, u = build_triangle_mesh_from_depth(
            depth=depth,
            intrinsics=intrinsics,
            stride=stride,
            min_depth=min_depth,
            max_depth=max_depth,
            edge_threshold_factor=edge_threshold_factor,
            extra_valid_mask=mask,
        )
        if len(t) > 0:
            out[k] = (v, t, u)
    return out
