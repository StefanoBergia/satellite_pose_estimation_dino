"""Test-Time Adaptation (TTA) engine for DINOv3 satellite pose estimation.

Implements three adaptation methods:
  1. norm_adapt  — update LayerNorm statistics from target domain
  2. tent        — entropy minimization on LayerNorm affine params (Wang et al., 2021)
  3. memo        — marginal entropy minimization with augmentations

All methods work with any existing checkpoint, no retraining needed.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utility: heatmap entropy
# ---------------------------------------------------------------------------

def heatmap_entropy(heatmaps: torch.Tensor) -> torch.Tensor:
    """Compute entropy of heatmap predictions.

    Args:
        heatmaps: (B, K, H, W) — already softmax-normalized probability maps

    Returns:
        (B,) mean entropy per sample across all keypoints
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)  # (B, K, H*W)
    # Clamp to avoid log(0)
    flat = flat.clamp(min=1e-8)
    ent = -(flat * flat.log()).sum(dim=-1)  # (B, K)
    return ent.mean(dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# Method 1: Norm Adaptation
# ---------------------------------------------------------------------------

def collect_layernorm_modules(model):
    """Find all LayerNorm modules in the ViT backbone."""
    ln_modules = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_modules.append((name, module))
    return ln_modules


class NormAdapt:
    """Adapt LayerNorm statistics by running forward passes on target data.

    For ViT LayerNorm (which doesn't track running stats like BatchNorm),
    we replace each LayerNorm with a version that uses batch statistics
    computed from the target domain.
    """

    def __init__(self, model):
        self.model = model
        self.original_state = None

    def adapt(self, loader, device, num_batches=None):
        """Run forward passes to collect target domain statistics.

        Replaces LayerNorm weight/bias with values fitted to target data:
        - Computes running mean/var of pre-norm activations
        - Adjusts LayerNorm affine params to re-center/re-scale
        """
        self.original_state = copy.deepcopy(self.model.state_dict())
        self.model.eval()

        # Collect output statistics for each LayerNorm via hooks
        ln_modules = collect_layernorm_modules(self.model)
        stats = {name: {"sum": None, "sq_sum": None, "count": 0}
                 for name, _ in ln_modules}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                x = output.detach()  # (B, N, D) or (B, D)
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])  # (B*N, D)
                s = stats[name]
                if s["sum"] is None:
                    s["sum"] = x.sum(dim=0)
                    s["sq_sum"] = (x ** 2).sum(dim=0)
                else:
                    s["sum"] += x.sum(dim=0)
                    s["sq_sum"] += (x ** 2).sum(dim=0)
                s["count"] += x.shape[0]
            return hook_fn

        for name, module in ln_modules:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        # Forward pass through target data
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc="NormAdapt: collecting stats")):
                if num_batches is not None and i >= num_batches:
                    break
                images = batch["image"].to(device)
                self.model(pixel_values=images)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Update LayerNorm parameters based on collected stats
        for name, module in ln_modules:
            s = stats[name]
            if s["count"] == 0:
                continue
            mean = s["sum"] / s["count"]
            var = s["sq_sum"] / s["count"] - mean ** 2
            # Adjust affine: new_weight = old_weight * old_std / new_std
            # new_bias = old_bias + old_weight * (old_mean - new_mean) / new_std
            # This is approximate; the key effect is re-centering
            std = (var + module.eps).sqrt()
            if module.weight is not None:
                module.bias.data += module.weight.data * (
                    -mean / std
                )

        print(f"NormAdapt: updated {len(ln_modules)} LayerNorm modules "
              f"using {stats[ln_modules[0][0]]['count']} tokens")

    def restore(self):
        """Restore original model parameters."""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state)
            self.original_state = None


# ---------------------------------------------------------------------------
# Method 2: TENT (Entropy Minimization)
# ---------------------------------------------------------------------------

class TENT:
    """Fully Test-Time Adaptation by Entropy Minimization (Wang et al., 2021).

    Only updates LayerNorm affine parameters (weight, bias).
    Minimizes the entropy of heatmap predictions on target domain images.
    """

    def __init__(self, model, lr=1e-4, num_steps=5):
        self.model = model
        self.lr = lr
        self.num_steps = num_steps
        self.original_state = None

    def _get_ln_params(self):
        """Get only LayerNorm affine parameters for optimization."""
        params = []
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    params.append(module.weight)
                if module.bias is not None:
                    params.append(module.bias)
        return params

    def adapt(self, loader, device, num_steps=None):
        """Adapt model by minimizing prediction entropy on target data."""
        if num_steps is None:
            num_steps = self.num_steps

        self.original_state = copy.deepcopy(self.model.state_dict())

        # Freeze everything except LayerNorm affine params
        for param in self.model.parameters():
            param.requires_grad = False

        ln_params = self._get_ln_params()
        for param in ln_params:
            param.requires_grad = True

        optimizer = torch.optim.Adam(ln_params, lr=self.lr)

        self.model.train()  # enable grad computation in LayerNorm

        step = 0
        total_entropy = 0.0
        n_batches = 0

        for _ in range(num_steps):
            for batch in loader:
                if step >= num_steps:
                    break
                images = batch["image"].to(device)

                model_out = self.model(pixel_values=images)

                if "heatmaps" not in model_out:
                    raise RuntimeError(
                        "TENT requires heatmap head (keypoint_head_type='heatmap')"
                    )

                # Entropy of heatmap predictions
                ent = heatmap_entropy(model_out["heatmaps"])
                loss = ent.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_entropy += loss.item()
                n_batches += 1
                step += 1

                if step >= num_steps:
                    break

        # Set back to eval mode
        self.model.eval()

        if n_batches > 0:
            print(f"TENT: {step} steps, avg entropy: {total_entropy / n_batches:.4f}")

    def restore(self):
        """Restore original model parameters."""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state)
            self.original_state = None


# ---------------------------------------------------------------------------
# Method 3: MEMO (Marginal Entropy Minimization with One test point)
# ---------------------------------------------------------------------------

class MEMO:
    """MEMO: adapt to each test sample using multiple augmentations.

    For each test image:
      1. Generate N augmented versions
      2. Forward all through the model
      3. Average heatmap predictions
      4. Minimize entropy of the average
      5. Update LayerNorm params with a few gradient steps
    """

    def __init__(self, model, lr=1e-4, num_augmentations=8, num_steps=1):
        self.model = model
        self.lr = lr
        self.num_augmentations = num_augmentations
        self.num_steps = num_steps
        self.original_state = None

    def _augment_batch(self, images):
        """Create augmented versions of the input images.

        Simple photometric augmentations that don't change geometry.
        """
        B, C, H, W = images.shape
        augmented = [images]  # include original

        for _ in range(self.num_augmentations - 1):
            aug = images.clone()
            # Random brightness
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                aug = (aug * factor).clamp(0, 1)
            # Random contrast
            if random.random() < 0.5:
                mean = aug.mean(dim=(2, 3), keepdim=True)
                factor = random.uniform(0.7, 1.3)
                aug = ((aug - mean) * factor + mean).clamp(0, 1)
            # Random Gaussian noise
            if random.random() < 0.5:
                noise = torch.randn_like(aug) * 0.02
                aug = (aug + noise).clamp(0, 1)
            augmented.append(aug)

        return torch.cat(augmented, dim=0)  # (B*N, C, H, W)

    def adapt_and_predict(self, images, device):
        """Adapt to a single batch and return predictions.

        Args:
            images: (B, C, H, W) input images

        Returns:
            dict with adapted model predictions
        """
        self.original_state = copy.deepcopy(self.model.state_dict())

        # Setup: only update LayerNorm params
        for param in self.model.parameters():
            param.requires_grad = False
        ln_params = []
        for module in self.model.backbone.modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    module.weight.requires_grad = True
                    ln_params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad = True
                    ln_params.append(module.bias)

        optimizer = torch.optim.Adam(ln_params, lr=self.lr)
        self.model.train()

        B = images.shape[0]

        # Adaptation steps
        for _ in range(self.num_steps):
            aug_images = self._augment_batch(images)  # (B*N, C, H, W)
            model_out = self.model(pixel_values=aug_images)

            if "heatmaps" not in model_out:
                raise RuntimeError("MEMO requires heatmap head")

            # Average heatmaps across augmentations
            hm = model_out["heatmaps"]  # (B*N, K, H, W)
            N = self.num_augmentations
            hm = hm.view(N, B, *hm.shape[1:])  # (N, B, K, H, W)
            avg_hm = hm.mean(dim=0)  # (B, K, H, W)

            # Minimize entropy of averaged predictions
            ent = heatmap_entropy(avg_hm)
            loss = ent.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final prediction with adapted model
        self.model.eval()
        with torch.no_grad():
            result = self.model(pixel_values=images)

        return result

    def restore(self):
        """Restore original model parameters."""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state)
            self.original_state = None
