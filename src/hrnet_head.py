"""HRNet keypoint head: simplified heatmap head for CNN spatial feature maps.

HRNet already outputs high-resolution feature maps at 1/4 input resolution
(56x56 for 224px input), so no deconvolution is needed. The head is a single
1x1 Conv2d producing per-keypoint heatmaps, followed by soft-argmax to decode
keypoint coordinates.

The soft-argmax logic is identical to HeatmapKeypointHead in model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRNetKeypointHead(nn.Module):
    """Keypoint head for HRNet backbone.

    Input:  (B, in_channels, H, W)  e.g. (B, 48, 56, 56) for 224px input
    Output: dict with:
        keypoints: (B, K, 2) in [0, 1] crop-relative coordinates
        heatmaps:  (B, K, H, W) spatial softmax probability maps
    """

    def __init__(
        self,
        in_channels: int = 48,
        num_keypoints: int = 11,
        heatmap_size: int = 56,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        # Single 1x1 conv: (B, C, H, W) -> (B, K, H, W)
        self.final_conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

        # Learnable temperature for soft-argmax (initialized same as HeatmapKeypointHead)
        self.beta = nn.Parameter(torch.tensor(float(heatmap_size)))

        # Precompute coordinate grids for soft-argmax
        x_coords = torch.linspace(0, 1, heatmap_size)
        y_coords = torch.linspace(0, 1, heatmap_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.register_buffer("grid_x", xx.reshape(1, 1, -1))  # (1, 1, H*W)
        self.register_buffer("grid_y", yy.reshape(1, 1, -1))  # (1, 1, H*W)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.final_conv.weight, std=0.001)
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, feat_map: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            feat_map: (B, C, H, W) HRNet high-res feature map

        Returns:
            dict with:
                keypoints: (B, K, 2) in [0, 1]
                heatmaps: (B, K, H, W) normalized probability maps
        """
        B = feat_map.shape[0]

        # Resize to target heatmap size if needed (handles non-224 input sizes)
        if feat_map.shape[-1] != self.heatmap_size or feat_map.shape[-2] != self.heatmap_size:
            feat_map = F.interpolate(
                feat_map,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        # Predict raw heatmap logits
        raw_heatmaps = self.final_conv(feat_map)  # (B, K, H, W)

        # Spatial softmax: convert logits to normalized probability maps
        K = self.num_keypoints
        flat = raw_heatmaps.view(B, K, -1)  # (B, K, H*W)
        weights = F.softmax(flat * self.beta, dim=-1)  # (B, K, H*W)
        heatmaps = weights.view(B, K, self.heatmap_size, self.heatmap_size)

        # Soft-argmax: weighted sum of coordinates
        coord_x = (weights * self.grid_x).sum(dim=-1)  # (B, K)
        coord_y = (weights * self.grid_y).sum(dim=-1)  # (B, K)
        coords = torch.stack([coord_x, coord_y], dim=-1)  # (B, K, 2)

        return {"keypoints": coords, "heatmaps": heatmaps}
