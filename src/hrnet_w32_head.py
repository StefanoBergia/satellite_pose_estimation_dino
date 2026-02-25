"""HRNet-W32 keypoint head: replicates colleague's exact architecture.

Architecture (must match checkpoint keys exactly):
  head.0: Conv2d(128, 256, 3, padding=1)
  head.1: BatchNorm2d(256)
  head.2: ReLU(inplace=True)
  head.3: Conv2d(256, 256, 3, padding=1)
  head.4: BatchNorm2d(256)
  head.5: ReLU(inplace=True)
  head.6: Dropout2d(0.1)
  final:  Conv2d(256, 11, 1)

Soft-argmax with fixed beta=50 (matching colleague's softargmax.py).
Output: keypoints in [0, 1] normalized coordinates (heatmap pixel coords
scaled by 1/(heatmap_size-1)), raw logit heatmaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRNetW32KeypointHead(nn.Module):
    """Keypoint head for HRNet-W32 backbone (colleague's architecture).

    Input:  (B, 128, H, W)  e.g. (B, 128, 128, 128) for 512px input
    Output: dict with:
        keypoints: (B, K, 2) in [0, 1] crop-relative coordinates
        heatmaps:  (B, K, H, W) spatial softmax probability maps
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_keypoints: int = 11,
        heatmap_size: int = 128,
        beta: float = 50.0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        # Colleague's exact head architecture (bias=False on Conv2d before BN)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),  # head.0
            nn.BatchNorm2d(256),                                     # head.1
            nn.ReLU(inplace=True),                                   # head.2
            nn.Conv2d(256, 256, 3, padding=1, bias=False),           # head.3
            nn.BatchNorm2d(256),                                     # head.4
            nn.ReLU(inplace=True),                                   # head.5
            nn.Dropout2d(0.1),                                       # head.6
        )
        self.final = nn.Conv2d(256, num_keypoints, 1)     # final

        # Fixed beta=50 (matching colleague's soft_argmax_2d)
        self.beta = beta

        # Coordinate grid in heatmap pixel space [0, H-1] (colleague's convention),
        # then normalized to [0, 1] by dividing by (H-1) for compatibility with
        # our evaluation pipeline's crop-relative coordinate mapping.
        x_coords = torch.linspace(0, heatmap_size - 1, heatmap_size) / (heatmap_size - 1)
        y_coords = torch.linspace(0, heatmap_size - 1, heatmap_size) / (heatmap_size - 1)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.register_buffer("grid_x", xx.reshape(1, 1, -1))  # (1, 1, H*W)
        self.register_buffer("grid_y", yy.reshape(1, 1, -1))  # (1, 1, H*W)

    def forward(self, feat_map: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            feat_map: (B, 128, H, W) HRNet-W32 feature map

        Returns:
            dict with:
                keypoints: (B, K, 2) in [0, 1]
                heatmaps: (B, K, H, W) raw logits
        """
        B = feat_map.shape[0]

        # Resize to target heatmap size if needed
        if feat_map.shape[-1] != self.heatmap_size or feat_map.shape[-2] != self.heatmap_size:
            feat_map = F.interpolate(
                feat_map,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        x = self.head(feat_map)           # (B, 256, H, W)
        raw_heatmaps = self.final(x)      # (B, K, H, W)

        # Spatial softmax with fixed beta (matching colleague's beta=50)
        K = self.num_keypoints
        flat = raw_heatmaps.view(B, K, -1)                    # (B, K, H*W)
        weights = F.softmax(flat * self.beta, dim=-1)          # (B, K, H*W)

        heatmaps = weights.view(B, K, self.heatmap_size, self.heatmap_size)

        # Soft-argmax: weighted sum of coordinates
        coord_x = (weights * self.grid_x).sum(dim=-1)         # (B, K)
        coord_y = (weights * self.grid_y).sum(dim=-1)         # (B, K)
        coords = torch.stack([coord_x, coord_y], dim=-1)      # (B, K, 2)

        return {"keypoints": coords, "heatmaps": heatmaps}
