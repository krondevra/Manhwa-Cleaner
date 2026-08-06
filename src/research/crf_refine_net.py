"""Halo fix, attempt 8 (`.claude/plans/snazzy-cuddling-creek.md` Part C): a differentiable
mean-field CRF refinement layer. Genuinely new signal source relative to all 7 prior mechanisms
(5 dense whole-page, 1 dense instance-scoped crop refiner (`halo_refiner.py`), 1 parametric
contour deformation (`contour_deform_net.py`)) -- every one of those transforms only the model's
OWN learned representation. This layer's pairwise term conditions directly on real image RGB
color, independent of any model confidence, per the halo occlusion probe's original finding that
the failure mode is a confident, context-INDEPENDENT "keep" prior baked into the model itself.

Windowed local approximation of a dense bilateral CRF (Krahenbuhl & Koltun 2011 "DenseCRF";
CRF-as-RNN, Zheng et al. 2015): real DenseCRF solves mean-field inference over ALL pixel pairs
via a permutohedral-lattice approximation; this uses a fixed local KxK neighborhood instead
(vectorized via `F.unfold`, not a full lattice) -- less expressive at long range, but the same
core mechanism (iterative mean-field updates combining a unary term with a learnable
spatial+color bilateral pairwise term) and tractable to implement/train from scratch on CPU here.

Drop-in I/O contract with `HaloRefinerNet` (`halo_refiner.py`): forward(x) where
x = (N, 4, H, W) = [rgb(3ch, 0..1), input_delete_mask(1ch)], returns (N, 1, H, W) logits. This
lets `train_crf_refine.py` reuse `HaloRefinerCropDataset`/`HaloRefinerRealErrorCropDataset` and
their dataset-pairs sourcing (100% synthetic P&C-generated pages, never real manhwa) unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFieldCRF(nn.Module):
    """One learnable bilateral mean-field CRF, windowed. Each iteration re-estimates pixel
    probability Q from the fixed unary term plus a weighted average of neighboring Q values,
    where the weight is exp(-spatial_dist^2/2*theta_alpha^2 - color_dist^2/2*theta_beta^2) --
    same functional form DenseCRF's bilateral kernel uses, restricted to a local window.
    theta_alpha/theta_beta (spatial/color bandwidth) and the unary<->pairwise compatibility
    weight are all learned, not fixed by hand.
    """

    def __init__(self, window: int = 5, n_iters: int = 3) -> None:
        super().__init__()
        assert window % 2 == 1, "window must be odd (has a center tap)"
        self.window = window
        self.n_iters = n_iters
        r = window // 2
        ys, xs = torch.meshgrid(
            torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij"
        )
        sp_sq = (ys.float() ** 2 + xs.float() ** 2).view(1, 1, window * window, 1)
        self.register_buffer("sp_sq", sp_sq)
        self.center_idx = (window * window) // 2

        # log-parameterized so theta_alpha/theta_beta stay positive under unconstrained SGD
        self.log_theta_alpha = nn.Parameter(torch.tensor(float(np.log(max(r, 1)))))
        self.log_theta_beta = nn.Parameter(torch.tensor(0.0))
        self.compat_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, unary_logits: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """unary_logits: (N,1,H,W). rgb: (N,3,H,W) in [0,1]. Returns Q, a PROBABILITY map
        (N,1,H,W), not logits -- caller converts back to logit space if needed."""
        n, _, h, w = unary_logits.shape
        r = self.window // 2
        theta_alpha = torch.exp(self.log_theta_alpha) + 1e-3
        theta_beta = torch.exp(self.log_theta_beta) + 1e-3

        rgb_pad = F.pad(rgb, [r, r, r, r], mode="replicate")
        rgb_unfold = F.unfold(rgb_pad, kernel_size=self.window)  # (N, 3*ww, H*W)
        rgb_unfold = rgb_unfold.view(n, 3, self.window * self.window, h * w)
        rgb_center = rgb.reshape(n, 3, 1, h * w)
        color_diff_sq = ((rgb_unfold - rgb_center) ** 2).sum(dim=1, keepdim=True)  # (N,1,ww,HW)

        weight = torch.exp(-self.sp_sq / (2 * theta_alpha**2) - color_diff_sq / (2 * theta_beta**2))
        weight = weight.clone()
        weight[:, :, self.center_idx, :] = 0.0  # a pixel is not its own neighbor
        weight_sum = weight.sum(dim=2, keepdim=False) + 1e-6  # (N,1,HW)

        Q = torch.sigmoid(unary_logits)
        for _ in range(self.n_iters):
            q_pad = F.pad(Q, [r, r, r, r], mode="replicate")
            q_unfold = F.unfold(q_pad, kernel_size=self.window)  # (N, ww, H*W)
            q_unfold = q_unfold.view(n, 1, self.window * self.window, h * w)
            pairwise = (weight * q_unfold).sum(dim=2) / weight_sum  # (N,1,HW)
            pairwise = pairwise.view(n, 1, h, w)
            # mean-field update: unary + compatibility-scaled pairwise "vote", re-squashed
            Q = torch.sigmoid(unary_logits + self.compat_weight * (2.0 * pairwise - 1.0))
        return Q


class CRFRefineNet(nn.Module):
    """Small unary CNN (few params, deliberately shallow -- the pairwise CRF term, not trunk
    depth, is this mechanism's actual bet) + MeanFieldCRF. Same (N,4,H,W)->(N,1,H,W)-logits
    contract as HaloRefinerNet."""

    def __init__(self, in_channels: int = 4, base: int = 16, window: int = 5, n_iters: int = 3) -> None:
        super().__init__()
        self.unary = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1),
        )
        self.crf = MeanFieldCRF(window=window, n_iters=n_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3, :, :]
        unary_logits = self.unary(x)
        q = self.crf(unary_logits, rgb).clamp(1e-6, 1 - 1e-6)
        return torch.log(q / (1 - q))  # back to logit space, for BCE-based losses/callers


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
