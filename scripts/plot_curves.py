"""Produce the report figures from convergence-study JSONL logs.

Figures:
    convergence_psnr.png    — Figure 2 (headline): PSNR vs iter, mean ± envelope.
    gradient_trajectory.png — Figure 3: gradient L2 norm vs iter.
    runtime_breakdown.png   — bar chart of time breakdown per iteration.
    iters_to_psnr_table.csv — Table 1 data for the LaTeX report.

Usage:
    python scripts/plot_curves.py
    python scripts/plot_curves.py --psnr-target 28.0 --outputs-dir data/outputs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path("data/outputs/figures")


def load_log(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_all_logs(outputs_dir: Path) -> dict[str, dict[str, Path]]:
    """Return {scene_id: {init_tag: log_path}} for every optimization run."""
    result: dict[str, dict[str, Path]] = {}
    for scene_dir in sorted(outputs_dir.iterdir()):
        opt_dir = scene_dir / "optimization"
        if not opt_dir.is_dir():
            continue
        for init_dir in sorted(opt_dir.iterdir()):
            if not init_dir.is_dir():
                continue
            tag = init_dir.name
            # Find log_{tag}.jsonl inside.
            log_path = init_dir / f"log_{tag}.jsonl"
            if log_path.exists():
                result.setdefault(scene_dir.name, {})[tag] = log_path
    return result


# ----------------------------------------------------------------------
# Figure 2: PSNR vs iteration
# ----------------------------------------------------------------------

def plot_psnr_convergence(
    logs_by_scene: dict[str, dict[str, Path]],
    out_path: Path,
    psnr_target: float = 28.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"depth": "#2b6cb0", "random": "#d53f8c"}
    labels = {"depth": "Depth-init (ours)", "random": "Random-init (3DGS baseline)"}

    # Aggregate across scenes for each init.
    curves_by_init: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for scene_id, init_map in logs_by_scene.items():
        for tag, path in init_map.items():
            base_init = tag.split("_")[0]  # strip '_depthreg'
            if base_init not in ("depth", "random"):
                continue
            if "depthreg" in tag:
                continue  # ablation goes in its own figure
            records = load_log(path)
            iters = np.array([r["iter"] for r in records])
            psnrs = np.array([r["psnr"] for r in records])
            curves_by_init.setdefault(base_init, []).append((iters, psnrs))

    for init, curves in curves_by_init.items():
        if not curves:
            continue
        # Interpolate onto common iter grid.
        common_iters = curves[0][0]
        matrix = np.stack(
            [np.interp(common_iters, x, y) for (x, y) in curves], axis=0,
        )
        mean = matrix.mean(axis=0)
        lo, hi = matrix.min(axis=0), matrix.max(axis=0)
        ax.plot(common_iters, mean, color=colors[init], label=labels[init], linewidth=2)
        ax.fill_between(common_iters, lo, hi, color=colors[init], alpha=0.18)

    ax.axhline(psnr_target, color="gray", linestyle="--", alpha=0.5,
               label=f"Target PSNR = {psnr_target}")
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Training-view PSNR (dB)")
    ax.set_title("Convergence: depth-init vs random-init")
    ax.set_xscale("symlog", linthresh=50)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------
# Figure 3: gradient trajectory
# ----------------------------------------------------------------------

def plot_gradient_trajectory(
    logs_by_scene: dict[str, dict[str, Path]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"depth": "#2b6cb0", "random": "#d53f8c"}
    labels = {"depth": "Depth-init (ours)", "random": "Random-init (baseline)"}

    by_init: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for scene_id, init_map in logs_by_scene.items():
        for tag, path in init_map.items():
            base_init = tag.split("_")[0]
            if base_init not in ("depth", "random") or "depthreg" in tag:
                continue
            records = load_log(path)
            iters = np.array([r["iter"] for r in records])
            grads = np.array([r["grad_norm_means"] for r in records])
            by_init.setdefault(base_init, []).append((iters, grads))

    for init, curves in by_init.items():
        if not curves:
            continue
        common = curves[0][0]
        matrix = np.stack(
            [np.interp(common, x, y) for (x, y) in curves], axis=0,
        )
        mean = matrix.mean(axis=0)
        lo, hi = np.percentile(matrix, [10, 90], axis=0)
        ax.plot(common, mean, color=colors[init], label=labels[init], linewidth=2)
        ax.fill_between(common, lo, hi, color=colors[init], alpha=0.15)

    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Gradient L2 norm on means")
    ax.set_title("Optimization-trajectory stability (H2)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------
# Table 1: iterations to reach target PSNR
# ----------------------------------------------------------------------

def iters_to_target_psnr(records: list[dict], target: float) -> int | None:
    """Return the first iteration where PSNR >= target, or None if never."""
    for r in records:
        if r["psnr"] >= target:
            return r["iter"]
    return None


def write_iters_table(
    logs_by_scene: dict[str, dict[str, Path]],
    out_path: Path,
    psnr_target: float = 28.0,
) -> None:
    rows = []
    speedups = []
    for scene_id, init_map in logs_by_scene.items():
        depth_records = load_log(init_map["depth"]) if "depth" in init_map else None
        random_records = (
            load_log(init_map["random"]) if "random" in init_map else None
        )
        depth_iter = (
            iters_to_target_psnr(depth_records, psnr_target) if depth_records else None
        )
        random_iter = (
            iters_to_target_psnr(random_records, psnr_target) if random_records else None
        )
        if depth_iter and random_iter:
            speedup = random_iter / depth_iter
            speedups.append(speedup)
        else:
            speedup = None
        rows.append([scene_id, depth_iter, random_iter, speedup])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "depth_init_iter", "random_init_iter", "speedup"])
        for row in rows:
            w.writerow([
                row[0],
                row[1] if row[1] is not None else "N/A",
                row[2] if row[2] is not None else "N/A",
                f"{row[3]:.2f}x" if row[3] is not None else "N/A",
            ])
        if speedups:
            w.writerow([
                "MEAN",
                "",
                "",
                f"{np.mean(speedups):.2f}x",
            ])

    print(f"Wrote {out_path}")
    print(f"  Mean speedup: {np.mean(speedups):.2f}x" if speedups else "  No data")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="data/outputs", type=Path)
    p.add_argument("--figure-dir", default="data/outputs/figures", type=Path)
    p.add_argument("--psnr-target", type=float, default=28.0)
    args = p.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    logs = find_all_logs(args.outputs_dir)

    if not logs:
        print("No optimization logs found.")
        print("Run: python scripts/run_convergence_study.py")
        return

    print(f"Found logs for {len(logs)} scene(s):")
    for scene_id, init_map in logs.items():
        print(f"  {scene_id}: {list(init_map.keys())}")

    plot_psnr_convergence(
        logs,
        args.figure_dir / "convergence_psnr.png",
        psnr_target=args.psnr_target,
    )
    plot_gradient_trajectory(
        logs,
        args.figure_dir / "gradient_trajectory.png",
    )
    write_iters_table(
        logs,
        args.figure_dir / "iters_to_psnr_table.csv",
        psnr_target=args.psnr_target,
    )

    print("\nSUGGESTION for your report:")
    print(f"  Figure 2: {args.figure_dir / 'convergence_psnr.png'}")
    print(f"  Figure 3: {args.figure_dir / 'gradient_trajectory.png'}")
    print(f"  Table 1:  {args.figure_dir / 'iters_to_psnr_table.csv'}")


if __name__ == "__main__":
    main()
