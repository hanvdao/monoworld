**Depth-Initialized Differentiable Gaussian Splatting from a Single Image**

**Han Dao** (`handao@stanford.edu`)

---

## Summary

The project asks: **how much does a good geometric prior buy you in the optimization of an explicit differentiable scene representation?** It answers this by building a single-image 3D scene reconstruction system that uses monocular depth estimation to produce a structured initialization for differentiable 3D Gaussian Splatting optimization. The system unprojects a single RGB photograph through a pinhole model using depth predictions, then converts the resulting colored points into a standard 3DGS `.ply` with depth-proportional scales (so each Gaussian projects to a consistent screen-space footprint regardless of its depth) and spherical-harmonic DC color coefficients. This scene then feeds a differentiable rasterizer, which runs Adam optimization over positions, log-scales, unit quaternions, opacity logits, and SH coefficients against a photometric loss. The system is implemented as a two-stage pipeline — a pretrained-model-driven initialization front-end, and a custom optimization loop with full systems instrumentation (per-iteration gradient norms, Gaussian counts, wall-clock timing) — so the contribution of the geometric prior can be isolated and measured quantitatively. The experimental comparison is initialized-depth-3DGS vs. [standard random-point-init 3DGS (Kerbl et al. 2023)](https://arxiv.org/abs/2308.04079) on 5 scenes.

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


