"""MonoWorld pipeline entrypoint.

Usage:
    python run.py --image data/inputs/example.jpg
    python run.py --image data/inputs/example.jpg --config configs/default.yaml
    python run.py --image data/inputs/example.jpg --preview
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from monoworld.depth import MiDaSDepthEstimator, DepthAnythingV2Estimator, colorize_depth
from monoworld.geometry import (
    auto_stride_for_budget,
    build_per_layer_meshes,
    build_triangle_mesh_from_depth,
    depth_to_colored_pointcloud,
    export_layered_glb,
    export_textured_glb,
    intrinsics_from_fov,
)
from monoworld.segmentation import (
    assign_depth_layers,
    colorize_layer_mask,
    layer_color_legend,
)
from monoworld.inpainting import inpaint_all_layers
from monoworld.scene import pointcloud_to_splat
from monoworld.utils import ensure_dir, get_logger, load_config, load_image, scene_id_from_path

log = get_logger("monoworld.run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MonoWorld end-to-end pipeline.")
    p.add_argument("--image", required=True, type=str, help="Path to input RGB image.")
    p.add_argument("--config", default="configs/default.yaml", type=str)
    p.add_argument("--scene-id", default=None, type=str,
                   help="Override auto scene id.")
    p.add_argument("--preview", action="store_true",
                   help="Open an Open3D window to preview the point cloud.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    image_path = Path(args.image)
    scene_id = args.scene_id or scene_id_from_path(image_path)
    out_dir = ensure_dir(Path(cfg["output"]["root"]) / scene_id)
    log.info("Scene id: %s", scene_id)
    log.info("Output dir: %s", out_dir)

    timings: dict[str, float] = {}

    # --- 1. Load image -------------------------------------------------------
    t0 = time.time()
    image = load_image(image_path)
    H, W = image.shape[:2]
    timings["load_image"] = time.time() - t0
    log.info("Loaded image: %dx%d", W, H)

    # --- 2. Depth inference --------------------------------------------------
    t0 = time.time()
    depth_model = cfg["depth"]["model"]
    if depth_model == "depth_anything_v2" and DepthAnythingV2Estimator is not None:
        log.info("Using Depth Anything V2")
        estimator = DepthAnythingV2Estimator(
            device=cfg["depth"]["device"],
            near=cfg["depth"]["near"],
            far=cfg["depth"]["far"],
        )
    else:
        if depth_model == "depth_anything_v2":
            log.warning("Depth Anything V2 unavailable (pip install transformers). "
                        "Falling back to MiDaS.")
        log.info("Using MiDaS DPT_Large")
        estimator = MiDaSDepthEstimator(
            device=cfg["depth"]["device"],
            near=cfg["depth"]["near"],
            far=cfg["depth"]["far"],
        )
    depth = estimator.infer(image)
    timings["depth_inference"] = time.time() - t0
    log.info("Depth map: shape=%s range=[%.3f, %.3f]",
             depth.shape, float(depth.min()), float(depth.max()))

    # --- 3. Save depth visualization ----------------------------------------
    if cfg["output"]["save_depth_vis"]:
        vis_bgr = colorize_depth(depth)
        vis_path = out_dir / "depth_vis.png"
        cv2.imwrite(str(vis_path), vis_bgr)
        np.save(out_dir / "depth_raw.npy", depth)
        log.info("Saved depth visualization: %s", vis_path)

    # --- 4. Intrinsics + point cloud ----------------------------------------
    t0 = time.time()
    intrinsics = intrinsics_from_fov(W, H, fov_deg=cfg["camera"]["fov_deg"])
    pcd = depth_to_colored_pointcloud(
        image_rgb=image,
        depth=depth,
        intrinsics=intrinsics,
        stride=cfg["pointcloud"]["stride"],
        min_depth=cfg["pointcloud"]["min_depth"],
        max_depth=cfg["pointcloud"]["max_depth"],
    )
    timings["pointcloud_build"] = time.time() - t0
    log.info("Point cloud: %d points", len(pcd.points))

    # --- 5. Save point cloud -------------------------------------------------
    if cfg["output"]["save_pointcloud"]:
        import open3d as o3d
        ply_path = out_dir / "pointcloud.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
        log.info("Saved point cloud: %s", ply_path)

    # --- 5b. Build + export textured mesh (Phase 3 + 4) ---------------------
    mesh_info: dict = {}
    layer_info: dict = {}
    if cfg["output"].get("save_mesh", True):
        t0 = time.time()
        mesh_cfg = cfg["mesh"]
        layers_cfg = cfg.get("layers", {"enabled": False})
        mesh_stride = auto_stride_for_budget(
            width=W, height=H,
            target_triangles=mesh_cfg["target_triangles"],
        )

        if layers_cfg.get("enabled", False):
            # Phase 4 path: assign layers, build one mesh per layer, export
            # multi-mesh .glb.
            layer_mask = assign_depth_layers(
                depth=depth,
                num_layers=layers_cfg["num_layers"],
                method=layers_cfg["method"],
                min_depth=mesh_cfg["min_depth"],
                max_depth=mesh_cfg["max_depth"],
            )
            if cfg["output"].get("save_layers", True):
                vis_bgr = colorize_layer_mask(layer_mask, layers_cfg["num_layers"])
                cv2.imwrite(str(out_dir / "layers.png"), vis_bgr)
                np.save(out_dir / "layers.npy", layer_mask)

            per_layer = build_per_layer_meshes(
                depth=depth,
                layer_mask=layer_mask,
                intrinsics=intrinsics,
                stride=mesh_stride,
                min_depth=mesh_cfg["min_depth"],
                max_depth=mesh_cfg["max_depth"],
                edge_threshold_factor=mesh_cfg["edge_threshold_factor"],
            )

            # --- Phase 5: inpaint disoccluded regions per layer -----
            inpaint_cfg = cfg.get("inpainting", {"enabled": False})
            per_layer_textures = None
            if inpaint_cfg.get("enabled", False):
                t1 = time.time()
                per_layer_textures = inpaint_all_layers(
                    image_rgb=image,
                    layer_mask=layer_mask,
                    num_layers=layers_cfg["num_layers"],
                    backend=inpaint_cfg.get("backend", "opencv"),
                    dilation_px=inpaint_cfg.get("dilation_px", 15),
                    prompt=inpaint_cfg.get("prompt", ""),
                    device=inpaint_cfg.get("device", "auto"),
                )
                timings["inpainting"] = time.time() - t1
                log.info("Inpainting complete (%.2fs, backend=%s)",
                         timings["inpainting"], inpaint_cfg.get("backend"))

                # Save debug outputs: inpaint masks + per-layer textures.
                from monoworld.inpainting.disocclusion import compute_layer_inpaint_mask
                inpaint_dir = ensure_dir(out_dir / "inpaint_debug")
                for k in range(layers_cfg["num_layers"]):
                    mask = compute_layer_inpaint_mask(
                        layer_mask, k, dilation_px=inpaint_cfg.get("dilation_px", 15),
                    )
                    cv2.imwrite(str(inpaint_dir / f"mask_layer_{k}.png"), mask)
                    if k in per_layer_textures:
                        tex_bgr = cv2.cvtColor(per_layer_textures[k], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(inpaint_dir / f"texture_layer_{k}.png"), tex_bgr)

            glb_path = out_dir / "mesh.glb"
            export_layered_glb(per_layer, image, glb_path,
                               per_layer_textures=per_layer_textures)

            total_v = sum(v.shape[0] for v, _, _ in per_layer.values())
            total_t = sum(t.shape[0] for _, t, _ in per_layer.values())
            mesh_info = {
                "stride": mesh_stride,
                "num_vertices": int(total_v),
                "num_triangles": int(total_t),
                "edge_threshold_factor": mesh_cfg["edge_threshold_factor"],
                "layered": True,
            }
            layer_info = {
                "num_layers": layers_cfg["num_layers"],
                "method": layers_cfg["method"],
                "color_legend_bgr": layer_color_legend(layers_cfg["num_layers"]),
                "per_layer_triangles": {
                    str(k): int(t.shape[0]) for k, (_, t, _) in per_layer.items()
                },
            }
            log.info("Layered mesh: %d layers, %d total tris (stride=%d) -> %s",
                     len(per_layer), total_t, mesh_stride, glb_path)

        else:
            # Phase 3 path: single mesh, no layering.
            vertices, triangles, uvs = build_triangle_mesh_from_depth(
                depth=depth,
                intrinsics=intrinsics,
                stride=mesh_stride,
                min_depth=mesh_cfg["min_depth"],
                max_depth=mesh_cfg["max_depth"],
                edge_threshold_factor=mesh_cfg["edge_threshold_factor"],
            )
            glb_path = out_dir / "mesh.glb"
            export_textured_glb(
                vertices=vertices, triangles=triangles, uvs=uvs,
                image_rgb=image, out_path=glb_path,
            )
            mesh_info = {
                "stride": mesh_stride,
                "num_vertices": int(vertices.shape[0]),
                "num_triangles": int(triangles.shape[0]),
                "edge_threshold_factor": mesh_cfg["edge_threshold_factor"],
                "layered": False,
            }
            log.info("Mesh: %d verts, %d tris (stride=%d) -> %s",
                     mesh_info["num_vertices"], mesh_info["num_triangles"],
                     mesh_stride, glb_path)

        timings["mesh_build_export"] = time.time() - t0

    # --- 5c. Gaussian Splatting export --------------------------------------
    splat_info: dict = {}
    if cfg["output"].get("save_splat", True):
        t0 = time.time()
        splat_cfg = cfg.get("splat", {})
        splat_stride = splat_cfg.get("stride", 2)
        splat_path = out_dir / "splat.ply"
        n_gaussians = pointcloud_to_splat(
            image_rgb=image,
            depth=depth,
            intrinsics=intrinsics,
            out_path=splat_path,
            stride=splat_stride,
            min_depth=cfg["pointcloud"]["min_depth"],
            max_depth=cfg["pointcloud"]["max_depth"],
            opacity=splat_cfg.get("opacity", 0.9),
        )
        timings["splat_export"] = time.time() - t0
        splat_info = {
            "num_gaussians": n_gaussians,
            "stride": splat_stride,
        }
        log.info("Gaussian splats: %d gaussians (stride=%d) -> %s",
                 n_gaussians, splat_stride, splat_path)

    # --- 6. Metadata ---------------------------------------------------------
    if cfg["output"]["save_metadata"]:
        meta = {
            "scene_id": scene_id,
            "input_image": str(image_path.resolve()),
            "image_size": {"width": W, "height": H},
            "intrinsics": {
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "cx": intrinsics.cx, "cy": intrinsics.cy,
                "fov_deg": cfg["camera"]["fov_deg"],
            },
            "depth": {
                "model": cfg["depth"]["model"],
                "near": cfg["depth"]["near"],
                "far": cfg["depth"]["far"],
                "min_raw": float(depth.min()),
                "max_raw": float(depth.max()),
            },
            "pointcloud": {
                "num_points": len(pcd.points),
                "stride": cfg["pointcloud"]["stride"],
            },
            "mesh": mesh_info,
            "layers": layer_info,
            "splat": splat_info,
            "timings_seconds": timings,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Saved metadata: %s", out_dir / "metadata.json")

    # --- 7. Update viewer scene index ---------------------------------------
    try:
        from scripts.build_scene_index import build_index
        idx_root = Path(cfg["output"]["root"])
        idx = build_index(idx_root)
        with open(idx_root / "scenes.json", "w") as f:
            json.dump(idx, f, indent=2)
        log.info("Updated scene index: %s (%d scenes)",
                 idx_root / "scenes.json", len(idx["scenes"]))
    except Exception as e:
        log.warning("Failed to update scene index: %s", e)

    log.info("Done. Total: %.2fs", sum(timings.values()))

    # --- 8. Optional local preview ------------------------------------------
    if args.preview:
        import open3d as o3d
        log.info("Opening Open3D preview. Close the window to exit.")
        o3d.visualization.draw_geometries([pcd], window_name=f"MonoWorld: {scene_id}")


if __name__ == "__main__":
    main()
