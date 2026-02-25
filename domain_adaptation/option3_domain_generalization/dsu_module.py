"""Domain Shifting Uncertainty (DSU) module for ViT.

Perturbs feature statistics (mean/std) with Gaussian noise during training
to simulate domain shifts, making the model more robust to distribution changes.

Reference: "Uncertainty Modeling for Out-of-Distribution Generalization"
           (Kang et al., ICLR 2022)
"""

import torch
import torch.nn as nn


class DSU(nn.Module):
    """Domain Shifting Uncertainty: perturbs feature statistics during training.

    During training:
        mu_perturbed = mu + eps1 * sigma,  eps1 ~ N(0, alpha)
        sigma_perturbed = sigma * (1 + eps2),  eps2 ~ N(0, beta)

    During eval: pass-through (no perturbation).

    Applied to features of shape (B, N, D) or (B, D) where D is the feature dim.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or (self.alpha == 0 and self.beta == 0):
            return x

        # Compute instance-level statistics
        if x.dim() == 4:
            # (B, C, H, W) — CNN feature map (e.g. HRNet)
            mu = x.mean(dim=(2, 3), keepdim=True)    # (B, C, 1, 1)
            sigma = x.std(dim=(2, 3), keepdim=True) + 1e-6  # (B, C, 1, 1)
        elif x.dim() == 3:
            # (B, N, D) — ViT sequence of tokens
            mu = x.mean(dim=1, keepdim=True)    # (B, 1, D)
            sigma = x.std(dim=1, keepdim=True) + 1e-6   # (B, 1, D)
        elif x.dim() == 2:
            # (B, D) — single vector
            mu = x.mean(dim=-1, keepdim=True)   # (B, 1)
            sigma = x.std(dim=-1, keepdim=True) + 1e-6  # (B, 1)
        else:
            return x  # unsupported shape, pass-through

        # Sample perturbation noise
        eps_mu = torch.randn_like(mu) * self.alpha
        eps_sigma = torch.randn_like(sigma) * self.beta

        # Perturb statistics
        mu_new = mu + eps_mu * sigma
        sigma_new = sigma * (1 + eps_sigma).clamp(min=0.1)

        # Normalize then re-apply perturbed statistics
        x_norm = (x - mu) / sigma
        return x_norm * sigma_new + mu_new

    def extra_repr(self):
        return f"alpha={self.alpha}, beta={self.beta}"
