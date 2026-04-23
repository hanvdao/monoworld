"""Run the full convergence study: all scenes × {depth-init, random-init}.

For each scene in data/outputs/ that has a splat.ply:
    - Optimize with depth-init.
    - Optimize with random-init.
    - (optional) Optimize with depth-init + depth-reg (H3 ablation).

Writes JSONL logs per run that plot_curves.py consumes.

Usage:
    python scripts/run_convergence_study.py
    python scripts/run_convergence_study.py --with-ablation
    python scripts/run_convergence_study.py --scenes scene1 scene2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="data/outputs", type=str)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Only these scene IDs (substring match). Default: all with splat.ply.",
    )
    p.add_argument(
        "--with-ablation",
        action="store_true",
        help="Also run depth-init + depth-reg (H3 ablation).",
    )
    p.add_argument(
        "--inits",
        nargs="*",
        default=["depth", "random"],
        choices=["depth", "random"],
    )
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    root = Path(args.outputs)
    all_scenes = [
        d for d in sorted(root.iterdir())
        if d.is_dir() and (d / "splat.ply").exists()
    ]
    if args.scenes:
        all_scenes = [
            s for s in all_scenes
            if any(pattern in s.name for pattern in args.scenes)
        ]

    if not all_scenes:
        print("No scenes found. Run `python run.py --image ...` first.")
        sys.exit(1)

    print(f"Convergence study over {len(all_scenes)} scenes:")
    for s in all_scenes:
        print(f"  {s.name}")
    print()

    for scene in all_scenes:
        for init in args.inits:
            print(f"\n{'=' * 70}")
            print(f"  {scene.name} :: init={init}")
            print(f"{'=' * 70}")
            cmd = [
                sys.executable,
                "optimize.py",
                "--scene", str(scene),
                "--init", init,
                "--iters", str(args.iters),
                "--device", args.device,
            ]
            subprocess.run(cmd, check=True)

        if args.with_ablation:
            print(f"\n{'=' * 70}")
            print(f"  {scene.name} :: init=depth + depth-reg (H3 ablation)")
            print(f"{'=' * 70}")
            cmd = [
                sys.executable,
                "optimize.py",
                "--scene", str(scene),
                "--init", "depth",
                "--depth-reg",
                "--iters", str(args.iters),
                "--device", args.device,
            ]
            subprocess.run(cmd, check=True)

    print("\n\nConvergence study complete. Next step:")
    print("  python scripts/plot_curves.py")


if __name__ == "__main__":
    main()
