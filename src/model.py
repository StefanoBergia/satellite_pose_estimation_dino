import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class KeypointHead(nn.Module):
    """MLP head that maps backbone features to keypoint coordinates."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        num_keypoints: int = 11,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_keypoints * 2))
        self.mlp = nn.Sequential(*layers)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) feature vector (CLS token)
        Returns:
            (B, num_keypoints, 2) predicted keypoint coords in [0, 1]
        """
        out = self.mlp(x)
        out = out.view(-1, self.num_keypoints, 2)
        out = torch.sigmoid(out)
        return out


class HeatmapKeypointHead(nn.Module):
    """Heatmap-based keypoint head using deconvolution on ViT patch tokens.

    Takes spatial patch tokens from the ViT backbone, reshapes them into a 2D
    feature map, upsamples via transposed convolutions, and predicts per-keypoint
    heatmaps. Coordinates are extracted via differentiable soft-argmax.
    """

    def __init__(
        self,
        in_dim: int,
        num_keypoints: int = 11,
        heatmap_size: int = 64,
        patch_grid_size: int = 14,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.patch_grid_size = patch_grid_size

        # Deconv upsampling: 14x14 -> 28x28 -> 56x56
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final 1x1 conv to predict per-keypoint heatmaps
        self.final_conv = nn.Conv2d(256, num_keypoints, kernel_size=1)

        # Learnable temperature for soft-argmax (initialized so softmax is sharp enough)
        self.beta = nn.Parameter(torch.tensor(float(heatmap_size)))

        # Precompute coordinate grids for soft-argmax
        x_coords = torch.linspace(0, 1, heatmap_size)
        y_coords = torch.linspace(0, 1, heatmap_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.register_buffer("grid_x", xx.reshape(1, 1, -1))  # (1, 1, H*W)
        self.register_buffer("grid_y", yy.reshape(1, 1, -1))  # (1, 1, H*W)

        self._init_weights()

    def _init_weights(self):
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.final_conv.weight, std=0.001)
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, num_patches, hidden_size) e.g. (B, 196, 1024)

        Returns:
            dict with:
                keypoints: (B, K, 2) in [0, 1]
                heatmaps: (B, K, H, W) raw logit heatmaps
        """
        B = patch_tokens.shape[0]
        H_p = W_p = self.patch_grid_size

        # Reshape to spatial feature map
        x = patch_tokens.transpose(1, 2).reshape(B, -1, H_p, W_p)  # (B, C, 14, 14)

        # Deconv upsample: 14 -> 28 -> 56
        x = self.deconv(x)  # (B, 256, 56, 56)

        # Bilinear upsample to target heatmap size
        if x.shape[-1] != self.heatmap_size:
            x = F.interpolate(
                x, size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )  # (B, 256, 64, 64)

        # Predict heatmaps (raw logits)
        raw_heatmaps = self.final_conv(x)  # (B, K, 64, 64)

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


class DifferentiablePnP(nn.Module):
    """Differentiable Perspective-n-Point layer.

    Given 2D keypoint predictions and known 3D model points, solves for the
    camera pose. Gradients flow back through the 2D keypoints.

    Uses a soft POSIT-like approach: iteratively solves for pose using a
    differentiable least-squares formulation.
    """

    def __init__(
        self,
        points_3d: torch.Tensor,
        camera_matrix: torch.Tensor,
        num_iterations: int = 10,
    ):
        """
        Args:
            points_3d: (K, 3) 3D model keypoints in object frame
            camera_matrix: (3, 3) camera intrinsic matrix
            num_iterations: number of iterative refinement steps
        """
        super().__init__()
        self.register_buffer("points_3d", points_3d)
        self.register_buffer("camera_matrix", camera_matrix)
        self.num_iterations = num_iterations

        # Precompute intrinsic params
        self.register_buffer("fx", camera_matrix[0, 0].unsqueeze(0))
        self.register_buffer("fy", camera_matrix[1, 1].unsqueeze(0))
        self.register_buffer("cx", camera_matrix[0, 2].unsqueeze(0))
        self.register_buffer("cy", camera_matrix[1, 2].unsqueeze(0))

    def forward(
        self,
        kp_2d: torch.Tensor,
        crop_box: torch.Tensor,
        img_size: torch.Tensor,
        visibility: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            kp_2d: (B, K, 2) predicted keypoints in crop-relative [0,1]
            crop_box: (B, 4) crop box (x1, y1, x2, y2) in pixels
            img_size: (B, 2) original image (w, h)
            visibility: (B, K) visibility flags

        Returns:
            dict with rotation (B, 3, 3) and translation (B, 3)
        """
        B, K, _ = kp_2d.shape

        # Convert crop-relative to full-image pixel coordinates
        x1 = crop_box[:, 0:1]  # (B, 1)
        y1 = crop_box[:, 1:2]
        x2 = crop_box[:, 2:3]
        y2 = crop_box[:, 3:4]
        crop_w = x2 - x1
        crop_h = y2 - y1

        kp_px = torch.zeros_like(kp_2d)
        kp_px[:, :, 0] = kp_2d[:, :, 0] * crop_w + x1
        kp_px[:, :, 1] = kp_2d[:, :, 1] * crop_h + y1

        # Normalize to camera coordinates: x_cam = (u - cx) / fx
        kp_cam = torch.zeros_like(kp_px)
        kp_cam[:, :, 0] = (kp_px[:, :, 0] - self.cx) / self.fx
        kp_cam[:, :, 1] = (kp_px[:, :, 1] - self.cy) / self.fy

        # Visibility mask
        mask = (visibility > 0).float()  # (B, K)

        # Solve pose via differentiable EPnP-like approach
        rotation, translation = self._solve_pose(kp_cam, mask)

        return {
            "rotation": rotation,
            "translation": translation,
        }

    def _solve_pose(
        self, kp_cam: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable pose estimation via weighted Procrustes on normalized coords.

        Args:
            kp_cam: (B, K, 2) keypoints in normalized camera coordinates
            mask: (B, K) visibility weights

        Returns:
            rotation: (B, 3, 3)
            translation: (B, 3)
        """
        B, K, _ = kp_cam.shape
        pts3d = self.points_3d.unsqueeze(0).expand(B, -1, -1)  # (B, K, 3)

        # Initial depth estimate: mean z of 3D points (rough)
        z_est = torch.ones(B, K, 1, device=kp_cam.device) * pts3d[:, :, 2:3].mean(dim=1, keepdim=True).clamp(min=0.5)

        for _ in range(self.num_iterations):
            # Construct 3D points in camera frame from 2D + estimated depth
            pts_cam = torch.cat([
                kp_cam[:, :, 0:1] * z_est,
                kp_cam[:, :, 1:2] * z_est,
                z_est,
            ], dim=-1)  # (B, K, 3)

            # Weighted centroids
            w = mask.unsqueeze(-1)  # (B, K, 1)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)

            centroid_cam = (pts_cam * w).sum(dim=1, keepdim=True) / w_sum  # (B, 1, 3)
            centroid_3d = (pts3d * w).sum(dim=1, keepdim=True) / w_sum

            # Center the points
            pts_cam_c = (pts_cam - centroid_cam) * w
            pts3d_c = (pts3d - centroid_3d) * w

            # SVD for rotation (Procrustes)
            H = torch.bmm(pts3d_c.transpose(1, 2), pts_cam_c)  # (B, 3, 3)
            U, S, Vh = torch.linalg.svd(H)

            # Ensure proper rotation (det = +1)
            det = torch.det(torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2)))
            sign = torch.ones(B, 3, device=kp_cam.device)
            sign[:, 2] = torch.sign(det)
            R = torch.bmm(Vh.transpose(1, 2) * sign.unsqueeze(1), U.transpose(1, 2))

            # Translation
            t = centroid_cam.squeeze(1) - torch.bmm(R, centroid_3d.squeeze(1).unsqueeze(-1)).squeeze(-1)

            # Update depth estimates: project 3D points with current R, t
            pts_proj = torch.bmm(pts3d, R.transpose(1, 2)) + t.unsqueeze(1)  # (B, K, 3)
            z_est = pts_proj[:, :, 2:3].clamp(min=0.1)

        return R, t


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert SPEED+ quaternion (w, x, y, z) to body-to-camera rotation matrix.

    SPEED+ stores q_vbs2tango as (w, x, y, z) scalar-first, representing the
    rotation from camera (VBS) frame to body (Tango) frame. quat2dcm gives
    R_cam2body; we transpose to get R_body2cam for projection.

    Args:
        q: (B, 4) quaternions in SPEED+ convention (w, x, y, z)

    Returns:
        (B, 3, 3) body-to-camera rotation matrices
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # DCM from camera to body (same as SPEED+ baseline quat2dcm)
    R_cam2body = torch.stack([
        2*w*w - 1 + 2*x*x,  2*x*y + 2*w*z,      2*x*z - 2*w*y,
        2*x*y - 2*w*z,      2*w*w - 1 + 2*y*y,  2*y*z + 2*w*x,
        2*x*z + 2*w*y,      2*y*z - 2*w*x,      2*w*w - 1 + 2*z*z,
    ], dim=-1).view(-1, 3, 3)

    # Transpose to get body-to-camera (needed for projection: P_cam = R @ P_body + t)
    return R_cam2body.transpose(1, 2)


class SatellitePoseModel(nn.Module):
    """Configurable model for satellite keypoint and pose estimation.

    Modes:
        - "keypoint_only": only keypoint regression head
        - "keypoint_pnp": keypoint + differentiable PnP (pose loss refines keypoints)
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        head_hidden_dims: list[int] | None = None,
        num_keypoints: int = 11,
        dropout: float = 0.1,
        mode: str = "keypoint_only",
        points_3d_path: str | None = None,
        camera_json_path: str | None = None,
        pnp_iterations: int = 10,
        keypoint_head_type: str = "mlp",
        heatmap_size: int = 64,
    ):
        super().__init__()

        if head_hidden_dims is None:
            head_hidden_dims = [512, 256]

        self.mode = mode
        self.num_keypoints = num_keypoints
        self.keypoint_head_type = keypoint_head_type

        # Load pretrained DINOv3 backbone
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 1024 for ViT-L

        if freeze_backbone:
            self.freeze_backbone(unfreeze_last_n=unfreeze_last_n_blocks)

        # Keypoint head (always present)
        if keypoint_head_type == "heatmap":
            patch_grid = self.backbone.config.image_size // self.backbone.config.patch_size
            self.keypoint_head = HeatmapKeypointHead(
                in_dim=hidden_size,
                num_keypoints=num_keypoints,
                heatmap_size=heatmap_size,
                patch_grid_size=patch_grid,
            )
        else:
            self.keypoint_head = KeypointHead(
                in_dim=hidden_size,
                hidden_dims=head_hidden_dims,
                num_keypoints=num_keypoints,
                dropout=dropout,
            )

        # Optional: differentiable PnP
        self.diff_pnp = None
        if mode == "keypoint_pnp":
            assert points_3d_path is not None, "points_3d_path required for PnP mode"
            assert camera_json_path is not None, "camera_json_path required for PnP mode"

            with open(points_3d_path, "r") as f:
                pts_data = json.load(f)
            points_3d = torch.tensor(pts_data["points"], dtype=torch.float32)

            with open(camera_json_path, "r") as f:
                cam_data = json.load(f)
            camera_matrix = torch.tensor(cam_data["cameraMatrix"], dtype=torch.float32)

            self.diff_pnp = DifferentiablePnP(
                points_3d=points_3d,
                camera_matrix=camera_matrix,
                num_iterations=pnp_iterations,
            )

    def freeze_backbone(self, unfreeze_last_n: int = 0):
        """Freeze backbone parameters, optionally unfreezing the last N transformer blocks.

        Args:
            unfreeze_last_n: Number of final transformer blocks to keep trainable.
                             Also unfreezes the final LayerNorm. 0 = fully frozen.
        """
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            # backbone.layer is nn.ModuleList of DINOv3ViTLayer (24 blocks for ViT-L)
            num_layers = len(self.backbone.layer)
            for layer in self.backbone.layer[num_layers - unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Also unfreeze the final LayerNorm (sits after the last block)
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        crop_box: torch.Tensor | None = None,
        img_size: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized image tensor
            crop_box: (B, 4) needed for PnP mode
            img_size: (B, 2) needed for PnP mode
            visibility: (B, K) needed for PnP mode

        Returns:
            dict with available outputs depending on mode
        """
        outputs = self.backbone(pixel_values=pixel_values)

        result = {}

        # Keypoints (always)
        if self.keypoint_head_type == "heatmap":
            # DINOv2/v3 layout: [CLS, reg1, ..., regN, patch_1, ..., patch_196]
            # Skip CLS + register tokens to get only the patch tokens
            num_patches = self.keypoint_head.patch_grid_size ** 2  # 196
            patch_tokens = outputs.last_hidden_state[:, -num_patches:]  # (B, 196, hidden_size)
            kp_out = self.keypoint_head(patch_tokens)
            result["keypoints"] = kp_out["keypoints"]
            result["heatmaps"] = kp_out["heatmaps"]
        else:
            cls_token = outputs.pooler_output  # (B, hidden_size)
            result["keypoints"] = self.keypoint_head(cls_token)

        # Differentiable PnP
        if self.diff_pnp is not None and crop_box is not None:
            pnp_out = self.diff_pnp(
                kp_2d=result["keypoints"],
                crop_box=crop_box,
                img_size=img_size,
                visibility=visibility,
            )
            result["pnp_rotation"] = pnp_out["rotation"]
            result["pnp_translation"] = pnp_out["translation"]

        return result
