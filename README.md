# MonoWorld: Monocular Image-to-Navigable 3D Scene Generation via Depth-Guided Layered Reconstruction

**Han Dao** (`handao@stanford.edu`)

---

## Summary

By the end of the quarter I will demo a system that takes a single RGB photograph as input and produces a navigable 3D scene rendered in a web browser, complete with WASD + mouse-look controls. The system will be evaluated across 3–5 diverse scenes (indoor, outdoor, close-up, wide-angle, stylized) by producing a side-by-side comparison of three render modes — colored point cloud, textured mesh with edge culling, and Gaussian splats — each built from the same depth-unprojected pipeline. The goal is to characterize the quality, runtime, and failure modes of each representation for the single-image-to-3D problem, both qualitatively (rendered stills + demo video) and quantitatively (PSNR / LPIPS at the origin view, FID under small novel-view offsets, and per-stage runtime). The approach combines a pretrained monocular depth estimator (Depth Anything V2), heuristic depth-based layering, OpenCV- and diffusion-based inpainting of disoccluded regions, and a custom GLSL shader for Gaussian splat rendering.

## Inputs and Outputs

**Inputs**
- A single RGB image (JPEG or PNG), typically 512×512 to 1280×720.
- An optional text prompt used to condition the diffusion-based inpainting stage (e.g., "sunny street, detailed, photorealistic").
- Assumed camera intrinsics derived from image size and a default horizontal FOV of 55°.

**Outputs**
- A Three.js-based browser viewer serving the generated scene at 60 FPS with real-time camera controls.
- Three switchable render representations per scene: point cloud (`.ply`), textured layered mesh (`.glb`), Gaussian splats (`.ply` in 3DGS format).
- Debug artifacts per scene: depth visualization, per-layer segmentation mask, per-layer inpainted textures, runtime metadata.
- A 2-minute demo video demonstrating navigation across multiple scenes and modes.

**Design constraints**
- **Consumer hardware** (Apple M-series Mac) is the target development platform. GPU memory limits forced the fallback from SDXL to SD 1.5 to OpenCV for inpainting; a stretch goal is to use cloud GPU credits to test higher-quality inpainters.
- **The output must be a real-time interactive demo** not offline renders. This rules out NeRF, which is competitive only with significant per-scene training time.

## Task List

### Nice-to-haves (if ahead of schedule)

- **Cloud-GPU-backed Flux Fill inpainting** for 1–2 showcase scenes. Flux Fill is the current state of the art for image inpainting (released Nov 2024 by Black Forest Labs) and would dramatically raise the quality ceiling on the disocclusion fill. Requires ~16 GB VRAM.
- **3DGS optimization pass.** Currently the splats are directly initialized from the depth-unprojected point cloud with no training. Running ~200 iterations of differentiable gradient descent against the source view would sharpen them significantly. Would require a GPU but minimal training time.

## Expected Deliverables and Evaluation

The evaluation framing follows the course's recommended "falsifiable hypothesis" structure.

**Hypothesis.** For single-image 3D scene generation targeting interactive browser-based navigation, the choice of scene representation (point cloud vs textured mesh vs Gaussian splats) drives perceived visual quality more than any individual component upgrade (depth model, inpainter). In particular, Gaussian splats will score measurably better on novel-view FID than textured meshes with inpainted disocclusion regions, *even without per-scene optimization*, because of their soft alpha falloff at silhouette edges.

**Primary evaluation: qualitative side-by-side comparison.** For each of 3–5 scenes, a figure containing three columns (Points / Mesh / Splats) × two rows (origin view / slightly-offset view). This directly demonstrates the representation trade-offs.

**Supporting quantitative metrics.**
- Reprojection PSNR + LPIPS at the origin view, per mode. (Expect: all three ≥ 28 dB PSNR since textures come from the source; LPIPS will show the meaningful differences.)
- FID of a small set (20) of novel-view renders against real reference photos. Small n, but directionally meaningful.
- Per-stage runtime table (depth inference, mesh build, inpainting, splat generation, viewer FPS).

**What success looks like:** a demo video and figure set where the viewer can clearly distinguish the three modes, with Splats visibly outperforming Mesh on soft transitions and Mesh visibly outperforming Points on surface solidity. Graphs showing how each stage contributes to quality, and an honest analysis of failure cases (thin foreground objects, reflective surfaces, very close camera translations).

## Risks and Mitigation

1. **GPU memory for diffusion inpainting.** Already bitten: SDXL and SD 1.5 both OOM'd at 1280×720 on my Mac. *Mitigated* by (a) fallback to OpenCV Telea (works everywhere), (b) resize-to-512 patch for the SD backend, (c) stretch-goal cloud GPU for the Flux upgrade.
3. **Single-image 3D reconstruction is fundamentally bounded.**  The project is explicitly about characterizing the ceiling of the depth-based approach not claiming to match multi-view diffusion systems.

## What I Need Help With

- **Advice on realistic FID evaluation with small n.** Is n=20 even publishable-grade, or should I drop FID entirely and just use PSNR+LPIPS+user-observed quality? Would appreciate a pointer to any novel-view eval protocol that's standard in the field.
- **A sanity-check on the "falsifiable hypothesis" framing** above before I commit to it for the report. Is this the level of specificity you're looking for?
- **Paper references for the related-work section**, specifically:
  - The canonical single-image-to-3D-scene paper that my approach is most similar to (I have in mind Shih et al. 2020 "3D Photography Using Context-aware Layered Depth Inpainting" but would welcome corrections).
  - A recent 3DGS-from-single-image paper to cite (LucidDreamer? ReconFusion?).
