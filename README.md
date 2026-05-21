# cs348k-project
**Depth-Initialized Differentiable Gaussian Splatting from a Single Image**

**Han Dao** (`handao@stanford.edu`)

---

## Summary

The project asks **how much does a good geometric prior buy you in the optimization of an explicit differentiable scene representation?** I attempt to this by building a single-image 3D scene reconstruction system that uses monocular depth estimation to produce a structured initialization for differentiable 3D Gaussian Splatting optimization. The system unprojects a single RGB photograph through a pinhole model using depth predictions, then converts the resulting colored points into a standard 3DGS `.ply` with depth-proportional scales (so each Gaussian projects to a consistent screen-space footprint regardless of its depth) and spherical-harmonic DC color coefficients. This scene then feeds a differentiable rasterizer, which runs Adam optimization over positions, log-scales, unit quaternions, opacity logits, and SH coefficients against a photometric loss. The system is implemented as a two-stage pipeline — a pretrained-model-driven initialization front-end, and a custom optimization loop with full systems instrumentation (per-iteration gradient norms, Gaussian counts, wall-clock timing) — so the contribution of the geometric prior can be isolated and measured quantitatively. The experimental comparison is initialized-depth-3DGS vs. [standard random-point-init 3DGS (Kerbl et al. 2023)](https://arxiv.org/abs/2308.04079) on 5 scenes.

## Inputs and Outputs

**Inputs**
- A single RGB image (512×512 to 1024×1024).
- Optimization config: iteration count, loss weights, densification schedule.

**Outputs**
- A trained 3DGS scene (`.ply`, standard 3DGS format).
- Convergence curves: PSNR / LPIPS vs iteration, both initialization strategies.
- Interactive browser viewer showing initialized vs optimized splats side-by-side.
- Per-stage runtime + memory profile of the optimization loop.
- Failure-case catalog with root-cause analysis.

**Design constraints**
- **Hardware:** might be CUDA-only, so optimization experiments run on a rented RunPod A10G 

## Task List
1. Reproducible end-to-end pipeline (`python run.py --image <path>`).
2. Depth estimation via Depth Anything V2 (primary) and MiDaS (fallback), config-selectable.
3. Unprojection to colored point cloud with synthesized pinhole intrinsics from a default 55° horizontal FOV.
4. Point cloud → 3DGS-format `.ply` with isotropic scales from local kNN density, SH DC coefficients, and sigmoid-logit opacities
5. Mesh baseline with edge-length culling --> which is the "naive depth-based reconstruction" the optimization has to beat.
6. Depth layering + per-layer inpainting for comparison context.
7. Browser viewer with a custom GLSL splat shader 
8. 25 unit tests across geometry, meshing, layering, inpainting, splat generation.
9. Differentiable rasterizer setup. 
10. Wrap the existing `.ply` splat generator as the initialization for the differentiable rasterizer.
11. Implement the optimization loop. 
13. Implement "random init" baseline from the 3DGS paper
14. Run the headline experiment on 5 scenes. For each scene, train both initializations for 2000 iterations
17. Systems analysis: profile the optimization loop. Forward rasterize vs backward vs densify vs loss. Produce a per-stage wall-clock breakdown.
18. Wire optimization output into the browser viewer so the demo toggles "initialized splats" ↔ "after N iterations" ↔ "fully optimized."
19. Record demo video showing the init → optimized progression + convergence curves.
20. Write report.

### Nice-to-haves (if ahead of schedule)

- Compare three init strategies in one plot: random, my depth-init, and a NeRF-style small-MLP-predicts-density warm-start. 
- Ship the custom PyTorch-MPS rasterizer. If I can get my own simplified rasterizer working well enough to produce the headline result on Mac without renting GPU, that becomes an additional systems contribution.

## Expected Deliverables and Evaluation
### Primary: convergence curves
- **Figure A (headline):** PSNR vs iteration, averaged over 5 scenes, two curves (depth-init vs random-init), shaded min/max envelope across scenes. 
- **Figure B:** Same as A but with LPIPS (perceptual).
- **Figure C:** Gradient-update magnitude vs iteration. 
- **Figure D:** Ablation table — {depth-init, random-init} × {with depth-reg, without}. 

### Supporting qualitative evaluation
Three-column comparison per scene: **original mesh baseline** (Phase 3) | **depth-init splats, no optimization** | **optimized splats after 2000 iters**. 


## Risks
1. **Single-image 3D reconstruction is fundamentally bounded.**  The project is explicitly about characterizing the ceiling of the depth-based approach not claiming to match multi-view diffusion systems.

2. **Densification is notoriously finicky.** If gradients explode or Gaussians collapse, the optimization diverges. Mitigation: start from the reference 3DGS schedule (densify every 100 iters, split if ||grad_position|| > τ, prune if opacity < α) and tune only if needed.
3. **Cloud GPU budget overrun.** 
4. **Single-image 3DGS may "overfit" trivially.** With only one training view, the model can memorize pixels while producing garbage geometry. 


## What I Need Help With
-Is this project advanced/substantial enough for the term project? 
- Is 6 scenes enough? Comparable papers use 8–20.
- Am I missing a canonical single-image-3DGS citation?


-----
# Checkpoint 1

This checkpoint focuses on the requirement that "evaluation code is running" For DepthInit-3DGS, the random-init 3DGS configuration is the trivial baseline — it's the standard initialization from Kerbl et al. 2023, and it gives my proposed depth-init something to be measured against. The full evaluation pipeline is implemented and has produced its first set of figures.

## What questions does the project aim to answer?
The central question: **how much does a good geometric prior buy you in the optimization of an explicit differentiable scene representation?**

The hypothesis structure: 
- **H1 (convergence acceleration):** depth-based init reduces iterations to a target PSNR by ≥5×
- **H2 (optimization trajectory):** depth-init produces smaller, more stable gradient updates
- **H3 (negative prediction):** depth-regularization during optimization gives no extra benefit in single-view

## What experiments answer the question?
A controlled comparison on the same scenes between two initialization strategies, both fed into the same Adam optimization loop with identical hyperparameters and densification schedule. Success at the checkpoint = "the experiment runs end-to-end and produces a falsifiable plot." Success at final report = "the plot supports or refutes the hypothesis with a quantitative claim."

## Status
All evaluation infrastructure works. Six scenes × two initializations × 2000 iterations completed on a RunPod RTX 2000 Ada. Logging, plotting, and analysis all run.

### H1: Convergence acceleration — **strongly supported**

| Threshold | Depth-init reaches at | Random-init reaches at | Speedup |
|---|---|---|---|
| PSNR=20 dB | mean iter 26 | mean iter ~917 (3 of 6 scenes never reach) | **263×** |
| PSNR=25 dB | mean iter ~80 | never reaches in 2000 iters | **∞** |
| PSNR=28 dB | mean iter 192 | never reaches in 2000 iters | **∞** |

At PSNR=20, depth-init reaches the threshold at iter 1 on 2 scenes, iter 50 on 3 scenes, and iter 300 on 1 scene. Random-init reaches PSNR=20 on only 3 of 6 scenes (at iter 250, 500, and 2000), and never reaches PSNR=25 on any scene. The headline plot shows random-init plateauing around PSNR=20 while depth-init climbs to PSNR=43+ by iter 2000.

H1 was originally framed as ">=5× speedup on >=4 of 5 scenes." The actual result far exceeds this: random-init does not produce a usable scene at all in this single-view setting within the iteration budget, while depth-init does so almost immediately.

### H2: Gradient-trajectory stability — **strongly supported**

Depth-init gradient L2 norm on means stays near `7e-3` for the entire training, with low variance across scenes. Random-init starts comparable but climbs to `5e-2` (~10× higher) by iter 1000 and stays there. The interpretation: depth-init Gaussians enter optimization in a low-curvature region of the loss landscape and refine smoothly, while random-init Gaussians spend most of training thrashing as they search for plausible geometry from scratch.

### H3: Depth-regularization during optimization — **not yet tested**

Ablation pending. Will run with the existing pipeline (`python optimize.py --depth-reg`), no new compute infrastructure needed.

## Where the code and results live

| Component | Path |
|---|---|
| Optimization module | `monoworld/optimization/{rasterizer,losses,densify,train_loop}.py` |
| Top-level entrypoint | `optimize.py` |
| Convergence study runner | `scripts/run_convergence_study.py` |
| Figure / table generator | `scripts/plot_curves.py` |
| Diagnostic tool | `scripts/diagnose_splat.py` |
| Headline figure | `data/outputs/figures/convergence_psnr.png` |
| Gradient-trajectory figure | `data/outputs/figures/gradient_trajectory.png` |
| Per-scene speedup table | `data/outputs/figures/iters_to_psnr_table.csv` |
| Per-run training logs | `data/outputs/<scene>/optimization/<init>/log_<init>.jsonl` |
| Optimized splat checkpoints | `data/outputs/<scene>/optimization/<init>/splat_<init>_final.ply` |

The figures and table can be regenerated end-to-end from the JSONL logs without re-running any training:

```bash
python scripts/plot_curves.py --psnr-target 20.0   # 263x mean speedup
python scripts/plot_curves.py --psnr-target 25.0   # depth-init reaches; random does not
python scripts/plot_curves.py --psnr-target 28.0   # depth-init reaches; random does not
```

## Reproducing the headline experiment

```bash
# Init pipeline (CPU/MPS/CUDA all work for this stage).
python run.py --image data/inputs/example.jpg

# Optimization (CUDA required for gsplat — use cloud GPU).
python optimize.py --scene data/outputs/<scene_id>/ --init depth   --iters 2000
python optimize.py --scene data/outputs/<scene_id>/ --init random  --iters 2000




# Plot.
python scripts/plot_curves.py --psnr-target 20.0
```


# Checkpoint 2

## 1. Solidified evaluation plan
After checkpoint 1, I made two changes in response to what the
data started showing:

- **Reframed the "speedup" metric.** checkpoint 1 reported speedup as "iterations to
  PSNR=20." That conflates two effects: depth-init being faster *per
  iteration* (fewer Gaussians early on) and depth-init *reaching the
  target at all* (random-init plateaus). I now report both an
  iteration-count speedup and a wall-clock speedup at three PSNR
  thresholds, with explicit "$\infty$ / never reached" entries instead of
  silently dropping non-converging scenes.

The final evaluation template, with status of each cell:
| Section | Cell | Status |
|---------|------|--------|
| H1 | PSNR vs iter curve (mean ± min-max over 6 scenes) | **filled** |
| H1 | Iters to PSNR={20, 25, 28} table | **filled** |
| H1 | Per-scene speedup breakdown | **filled** |
| H1 | Wall-clock time to PSNR={20, 25, 28} table | **filled**|
| H2 | Gradient L2-norm trajectory plot | **filled** |
| H3 | 2×2 ablation table (init × depth-reg) | **stubbed**; framework runs |
| Qualitative | 3-scene grid (init / depth-opt / random-opt) | **filled** |
| Qualitative | Off-axis oblique-view 3D-structure comparison | **filled** |
| Systems | Wall-clock per init, per PSNR threshold | **filled** (from CP2 work) |
| Systems | Per-stage timing breakdown | not started; needs re-instrumentation |
| Systems | Memory vs Gaussian count | not started; needs re-instrumentation |
| Failure cases | Three failure modes with one figure each | filled (Random-init iter progression captured) |

## 2. What was filled in since checkpoint 1
### H1 + H2 already complete at checkpoint 1
Carried forward without change. Headline result is unchanged: 263×
mean iteration speedup at PSNR=20, infinite speedup at PSNR=25/28.

### Wall-clock systems profile (new since CP1)
Built `scripts/profile.py` to aggregate the existing JSONL logs into a
wall-clock table. The JSONL logs already had `wall_seconds`, so no
rerun was needed.

| Init | t→PSNR=20 | t→PSNR=25 | t→PSNR=28 | Total 2000 iters | Mean per iter |
|------|-----------|-----------|-----------|------------------|---------------|
| Depth-init | 3.2 s | 9.6 s | 14.5 s | 2.1 min | 63 ms |
| Random-init | 1.5 min | 1.5 min (1 of 6 scenes) | never | 2.7 min | 82 ms |

Two findings here that were not visible from iteration-count alone:

- **Random-init does reach PSNR=25 on one scene** (test_4), which
  contradicts the checkpoint 1 report's "never reaches PSNR=25 on any scene"
  claim. The H1 table will be updated to "1 of 6" instead of "0 of 6"
  for that row.

- **Random-init is 30% slower per iteration** (82 vs 63 ms) despite
  ending with *fewer* Gaussians (198k vs 448k). The reason: depth-init's
  Gaussians settle quickly, so densification calls are cheap; random-init's
  Gaussians keep oscillating, so the rasterizer pays for more
  per-iteration overhead on Gaussians that get pruned in the next densify
  pass.

### Gaussian-count differential (new since checkpoint 1)
Depth-init ends at 448k Gaussians (mean over 6 scenes; 895k on the most
complex scene). Random-init plateaus at ~198k — essentially its initial
count. This is downstream of the structural failure. with no real
geometry, densification's clone/split rule has no high-gradient
Gaussians to expand from.

### Qualitative figures (new since checkpoint 1)
I captured screenshots of three scenes through the integrated
viewer (variant dropdown locked to same camera within a row): init,
depth-init optimized, and random-init optimized. Plus the off-axis
oblique-view figure showing depth-init preserving depth ordering vs
random-init collapsing to a planar sheet.

### Viewer integration
The Three.js viewer now has a variant dropdown that toggles between
{init, depth-opt, random-opt, intermediate checkpoints} without
moving the camera. This is what made the qualitative figure capture
practical — without it, getting matching camera poses across variants
was hand-eye coordination at best.

### Failure-case documentation
Three failure modes identified and written up with a supporting figure
for each:

1. **Random-init degenerate convergence** — flat-poster solution
   visible from any off-axis angle. Iter-500 / iter-1000 / iter-2000
   checkpoints all look structurally identical.
2. **Thin-structure dropout in depth-init** — overhead wires and
   distant tree branches are smoothed by Depth Anything V2 and the
   photometric loss cannot recover them.
3. **Off-axis viewpoint degradation** — fundamental limit of
   single-view reconstruction. Geometry in occluded regions is
   guessed by the depth model and never verified against image data.

### Report draft
I have drafted the report with Abstract, Introduction,
Methods, Evaluation (H1/H2/H3/qualitative/systems/failures),
Discussion, Appendix. All figures referenced exist on disk except for
the H3 table, which is stubbed for the ablation that has not yet been
run.

## 3. What's left to do
### Fill in stubbed evaluation cells

- **H3 ablation.** 2×2 table of {depth-init, random-init} × {with depth-reg,
  without depth-reg}, on the same 6 scenes for 2000 iters each. Existing
  pipeline supports this via the `--depth-reg` flag; needs ~$0.50 of GPU
  time and ~2 hours to run and aggregate.

- **Update H1 Table 2 row for PSNR=25.** Change from "0 of 6 scenes
  reach" to "1 of 6 (test_4 at ~iter 1100)." Already identified by the
  new wall-clock profile; just needs the LaTeX edit.

- **Per-stage timing breakdown** (forward / backward / loss / optimizer /
  densify). The current JSONL logger doesn't record per-stage timings.
  Filling this in requires re-instrumenting `train_loop.py` and rerunning
  one representative scene. Marginal value vs the cost of another GPU
  session — the wall-clock table already captures the systems story.
  Will be acknowledged as a limitation in the report.

- **Per-iter GPU memory.** Same situation as per-stage timing. The
  ~3–4 GB peak claim in the report is from `nvidia-smi` observation
  during the convergence study, not from per-iter logging. Acknowledged
  as a limitation.

### Writeup polish
- Figure caption tightening.
- Finish writing the report.
