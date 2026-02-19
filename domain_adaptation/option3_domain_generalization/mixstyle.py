"""MixStyle: Domain Generalization via feature-level style mixing.

Randomly mixes feature statistics (mean/std) between samples in a batch
during training. This teaches the model that content matters, not style.

Reference: "Domain Generalization with MixStyle" (Zhou et al., ICLR 2021)
"""

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle: mix instance-level feature statistics between batch samples.

    During training (with probability p):
        1. Compute per-instance mean/std
        2. Sample lambda ~ Beta(alpha, alpha)
        3. Shuffle instances to get random pairs
        4. Mix statistics: mu_mix = lambda*mu_i + (1-lambda)*mu_j
        5. Re-apply mixed statistics

    During eval: pass-through.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.shape[0]
        if B < 2:
            return x  # need at least 2 samples to mix

        if x.dim() == 3:
            # (B, N, D)
            mu = x.mean(dim=1, keepdim=True)    # (B, 1, D)
            sigma = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, D)
        elif x.dim() == 2:
            # (B, D)
            mu = x.mean(dim=-1, keepdim=True)
            sigma = x.std(dim=-1, keepdim=True) + self.eps
        else:
            return x

        # Normalize
        x_norm = (x - mu) / sigma

        # Sample mixing coefficient
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1) if x.dim() == 3 else (B, 1)
        ).to(x.device)

        # Random permutation for pairing
        perm = torch.randperm(B)

        # Mix statistics
        mu_mix = lmda * mu + (1 - lmda) * mu[perm]
        sigma_mix = lmda * sigma + (1 - lmda) * sigma[perm]

        return x_norm * sigma_mix + mu_mix

    def extra_repr(self):
        return f"p={self.p}, alpha={self.alpha}"
