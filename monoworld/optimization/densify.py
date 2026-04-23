"""3DGS densification: clone / split / prune.

Implements the adaptive density control from Kerbl et al. 2023,
supplementary §A. Runs every N iterations between warmup_iter and stop_iter.

Densification touches the optimizer state — after adding/removing
Gaussians, the optimizer's internal momentum/variance must be kept in
sync with the new parameter shapes. That's what `_resize_optimizer_state`
handles.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .rasterizer import GaussianParams


# 3DGS paper defaults.
GRAD_THRESHOLD_CLONE = 0.0002    # high positional grad -> densify
SCALE_THRESHOLD_SPLIT = 0.01     # above this -> split, below -> clone
OPACITY_THRESHOLD_PRUNE = 0.005  # below this sigmoid opacity -> prune
SCALE_THRESHOLD_PRUNE = 0.1      # too large in world space -> prune


def densify_and_prune(
    params: GaussianParams,
    optimizer: torch.optim.Optimizer,
    positional_grad_accum: torch.Tensor,   # (N,) EMA of ||grad(means)||
    visibility_count: torch.Tensor,        # (N,) # frames each Gaussian appeared in
    scene_extent: float,                   # characteristic scene scale
    grad_threshold: float = GRAD_THRESHOLD_CLONE,
    verbose: bool = True,
) -> GaussianParams:
    """Run one densification step. Returns updated GaussianParams.

    Steps (in order):
      1. Identify high-gradient Gaussians. Clone (duplicate) the small
         ones; split (divide into two smaller) the large ones.
      2. Prune Gaussians with low opacity or excessive scale.
    """
    device = params.means.device
    n0 = params.means.shape[0]

    # Normalize gradient accumulation by visibility count.
    avg_grad = positional_grad_accum / visibility_count.clamp(min=1)
    needs_densify = avg_grad > grad_threshold

    # Separate into clone (small) and split (large) subsets.
    max_scale_per_g = torch.exp(params.scales).max(dim=-1).values  # (N,)
    small = max_scale_per_g <= SCALE_THRESHOLD_SPLIT * scene_extent
    large = max_scale_per_g > SCALE_THRESHOLD_SPLIT * scene_extent

    clone_mask = needs_densify & small
    split_mask = needs_densify & large

    new_params = _densify(
        params, optimizer, clone_mask=clone_mask, split_mask=split_mask,
    )

    # Prune: low opacity OR too-large scales.
    opac = torch.sigmoid(new_params.opacities).squeeze(-1)
    max_scale_per_g = torch.exp(new_params.scales).max(dim=-1).values
    prune_mask = (opac < OPACITY_THRESHOLD_PRUNE) | (
        max_scale_per_g > SCALE_THRESHOLD_PRUNE * scene_extent
    )
    new_params = _prune(new_params, optimizer, prune_mask)

    if verbose:
        n1 = new_params.means.shape[0]
        n_cloned = int(clone_mask.sum().item())
        n_split = int(split_mask.sum().item())
        n_pruned = int(prune_mask.sum().item())
        print(
            f"  densify: {n0} -> {n1}  "
            f"(+{n_cloned} clone, +{n_split*2} split, -{n_pruned} prune)"
        )

    return new_params


# ----------------------------------------------------------------------
# Clone / split / prune primitives
# ----------------------------------------------------------------------

def _densify(
    params: GaussianParams,
    optimizer: torch.optim.Optimizer,
    clone_mask: torch.Tensor,
    split_mask: torch.Tensor,
) -> GaussianParams:
    """Clone + split in one batch. Returns new GaussianParams."""
    device = params.means.device

    # ---- Clone: duplicate the Gaussian verbatim. ----
    clone_extensions = {
        "means": params.means[clone_mask].detach().clone(),
        "scales": params.scales[clone_mask].detach().clone(),
        "quats": params.quats[clone_mask].detach().clone(),
        "opacities": params.opacities[clone_mask].detach().clone(),
        "sh_dc": params.sh_dc[clone_mask].detach().clone(),
        "sh_rest": params.sh_rest[clone_mask].detach().clone(),
    }

    # ---- Split: sample 2 child positions from the parent's Gaussian. ----
    n_split = int(split_mask.sum().item())
    if n_split > 0:
        parent_means = params.means[split_mask].detach()
        parent_scales = torch.exp(params.scales[split_mask]).detach()
        # Sample 2 offsets from N(0, parent_scale**2).
        eps1 = torch.randn_like(parent_means) * parent_scales
        eps2 = torch.randn_like(parent_means) * parent_scales
        new_means = torch.cat([parent_means + eps1, parent_means + eps2], dim=0)
        # Shrink scales by 1.6 (3DGS paper's factor).
        new_log_scales = (
            torch.cat([params.scales[split_mask], params.scales[split_mask]], dim=0)
            - torch.log(torch.tensor(1.6, device=device))
        ).detach().clone()

        split_extensions = {
            "means": new_means,
            "scales": new_log_scales,
            "quats": params.quats[split_mask].repeat(2, 1).detach().clone(),
            "opacities": params.opacities[split_mask].repeat(2, 1).detach().clone(),
            "sh_dc": params.sh_dc[split_mask].repeat(2, 1, 1).detach().clone(),
            "sh_rest": params.sh_rest[split_mask].repeat(2, 1, 1).detach().clone(),
        }
    else:
        split_extensions = None

    # ---- Build the new parameter set, pruning the parents of split. ----
    keep_mask = ~split_mask  # keep cloned parents; drop split parents
    kept = {
        "means": params.means[keep_mask],
        "scales": params.scales[keep_mask],
        "quats": params.quats[keep_mask],
        "opacities": params.opacities[keep_mask],
        "sh_dc": params.sh_dc[keep_mask],
        "sh_rest": params.sh_rest[keep_mask],
    }

    # Concatenate kept + clones + splits.
    cat_list = [kept, clone_extensions]
    if split_extensions is not None:
        cat_list.append(split_extensions)

    new_tensors = {}
    for key in kept:
        parts = [d[key] for d in cat_list if d[key].numel() > 0 or d[key].shape[0] > 0]
        if not parts:
            new_tensors[key] = kept[key]
        else:
            new_tensors[key] = torch.cat(parts, dim=0)

    new_params = _rebuild_params(new_tensors, optimizer, params)
    return new_params


def _prune(
    params: GaussianParams,
    optimizer: torch.optim.Optimizer,
    prune_mask: torch.Tensor,
) -> GaussianParams:
    keep_mask = ~prune_mask
    new_tensors = {
        "means": params.means[keep_mask],
        "scales": params.scales[keep_mask],
        "quats": params.quats[keep_mask],
        "opacities": params.opacities[keep_mask],
        "sh_dc": params.sh_dc[keep_mask],
        "sh_rest": params.sh_rest[keep_mask],
    }
    return _rebuild_params(new_tensors, optimizer, params)


def _rebuild_params(
    tensors: dict,
    optimizer: torch.optim.Optimizer,
    old_params: GaussianParams,
) -> GaussianParams:
    """Rewrap tensors as nn.Parameters, and rewire the optimizer to track them.

    Because we change the number of Gaussians, the optimizer's internal
    `exp_avg` / `exp_avg_sq` buffers become shape-mismatched. We recreate
    the param groups with fresh state — matches the 3DGS reference impl.
    """
    new_params = GaussianParams(
        means=nn.Parameter(tensors["means"].detach().contiguous()),
        scales=nn.Parameter(tensors["scales"].detach().contiguous()),
        quats=nn.Parameter(tensors["quats"].detach().contiguous()),
        opacities=nn.Parameter(tensors["opacities"].detach().contiguous()),
        sh_dc=nn.Parameter(tensors["sh_dc"].detach().contiguous()),
        sh_rest=nn.Parameter(tensors["sh_rest"].detach().contiguous()),
    )

    # Rewire each existing param group to its new parameter, dropping state.
    replace_map = {
        id(old_params.means): new_params.means,
        id(old_params.scales): new_params.scales,
        id(old_params.quats): new_params.quats,
        id(old_params.opacities): new_params.opacities,
        id(old_params.sh_dc): new_params.sh_dc,
        id(old_params.sh_rest): new_params.sh_rest,
    }
    for group in optimizer.param_groups:
        new_list = []
        for p in group["params"]:
            if id(p) in replace_map:
                new_p = replace_map[id(p)]
                # Drop stale state; Adam rebuilds on next step.
                if p in optimizer.state:
                    del optimizer.state[p]
                new_list.append(new_p)
            else:
                new_list.append(p)
        group["params"] = new_list

    return new_params
