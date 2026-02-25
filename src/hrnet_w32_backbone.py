"""HRNet-W32 backbone loader for satellite pose estimation.

Mirrors hrnet_backbone.py but for the W32 variant. Uses out_index=1 which
gives 128-channel stride-4 features (raw stage4 branch1 at 1/4 resolution).
"""

import torch
import torch.nn as nn


# Output channels from the stride-4 HRNet-W32 feature map.
# With feature_location='', features[1] is 128ch at stride-4.
HRNET_W32_OUT_CHANNELS = 128

# Which features_only index to use (stride-4 = index 1)
HRNET_W32_FEATURE_IDX = 1


def build_hrnet_w32(pretrained_path: str | None = None) -> nn.Module:
    """Load HRNet-W32 via timm.

    Args:
        pretrained_path: Path to a .pth checkpoint with backbone weights.
                         If None, uses timm's ImageNet pretrained weights.

    Returns:
        timm HRNet-W32 model with weights loaded.
    """
    import timm

    # Match colleague's exact timm call: out_indices=(1,) requests only the
    # stride-4 feature map (128ch). Default returns incre_modules bottleneck outputs.
    # Only load timm pretrained weights if no custom checkpoint is provided.
    model = timm.create_model("hrnet_w32", pretrained=(pretrained_path is None),
                              features_only=True, out_indices=(1,))

    if pretrained_path is not None:
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        backbone_state = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                new_key = k[len("backbone."):]
                backbone_state[new_key] = v

        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        n_loaded = len(backbone_state) - len(unexpected)
        print(f"HRNet-W32: loaded {n_loaded}/{len(backbone_state)} backbone keys "
              f"from {pretrained_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

    return model


def hrnet_w32_forward(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run HRNet-W32 forward pass and return the stride-4 feature map.

    Uses default feature_location (incre_modules bottleneck), so index [1]
    gives 128ch at stride-4. For 512px input: (B, 128, 128, 128).

    Args:
        backbone: timm HRNet-W32 model (features_only=True)
        x: (B, 3, H, W) input image tensor

    Returns:
        (B, 128, H/4, W/4) stride-4 feature map
    """
    features = backbone(x)
    # With out_indices=(1,), only one feature map is returned at index 0
    return features[0]


def freeze_hrnet_w32(backbone: nn.Module, unfreeze_last_n_stages: int = 0) -> None:
    """Freeze HRNet-W32 backbone, optionally keeping last N stages trainable."""
    for param in backbone.parameters():
        param.requires_grad = False

    if unfreeze_last_n_stages > 0:
        stage_names = ["layer1", "stage2", "stage3", "stage4"]
        transition_names = ["transition1", "transition2", "transition3"]

        stages_to_unfreeze = stage_names[-unfreeze_last_n_stages:]
        transitions_to_unfreeze = transition_names[-(unfreeze_last_n_stages - 1):] if unfreeze_last_n_stages > 1 else []

        for name in stages_to_unfreeze + transitions_to_unfreeze:
            module = getattr(backbone, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

        if unfreeze_last_n_stages >= 4:
            for stem_name in ["conv1", "bn1", "conv2", "bn2"]:
                module = getattr(backbone, stem_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True


def unfreeze_hrnet_w32(backbone: nn.Module) -> None:
    """Unfreeze all HRNet-W32 backbone parameters."""
    for param in backbone.parameters():
        param.requires_grad = True


def get_hrnet_w32_norm_modules(backbone: nn.Module) -> list[tuple[str, nn.Module]]:
    """Get all BatchNorm2d modules in the HRNet-W32 backbone."""
    return [
        (name, module)
        for name, module in backbone.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    ]
