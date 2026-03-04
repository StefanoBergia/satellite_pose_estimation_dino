"""DANN modules: Gradient Reversal Layer and 3-class Domain Classifier."""

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity in forward pass; negates and scales gradients in backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    """Gradient Reversal Layer — no learnable parameters."""

    def forward(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        return GradientReversalFunction.apply(x, lambda_)


class DomainClassifier(nn.Module):
    """3-class domain classifier: synthetic (0) / lightbox (1) / sunlamp (2).

    Applied on top of GRL-reversed backbone features to make the backbone
    domain-invariant via adversarial training.
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, num_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) backbone feature vector (CLS token)
        Returns:
            (B, num_domains) domain logits
        """
        return self.net(x)
