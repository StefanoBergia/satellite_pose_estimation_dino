"""HRNet-W48 backbone loader for satellite pose estimation.

Loads HRNet-W48 via timm, optionally initializing from an mmpose checkpoint.
The mmpose HRNet-W48 COCO keypoint checkpoint stores backbone weights under
the 'backbone.' prefix — these are stripped and loaded into the timm model.

If no pretrained_path is given, timm's built-in ImageNet pretrained weights
are used (downloaded automatically on first use).
"""

import torch
import torch.nn as nn


# Output channels from the stride-4 (56x56 for 224px input) HRNet-W48 feature map.
# With feature_location='', features[1] is raw stage4 branch0 = 48ch (not the 128ch
# incre_modules bottleneck used for classification).
HRNET_W48_OUT_CHANNELS = 48

# Which features_only index to use (stride-4 = index 1)
HRNET_FEATURE_IDX = 1


def build_hrnet_w48(pretrained_path: str | None = None) -> nn.Module:
    """Load HRNet-W48 via timm.

    Args:
        pretrained_path: Path to an mmpose HRNet-W48 .pth checkpoint.
                         If None, uses timm's ImageNet pretrained weights.

    Returns:
        timm HRNet-W48 model with weights loaded.
    """
    import timm  # imported lazily so DINOv3 path doesn't need timm installed

    use_timm_pretrained = pretrained_path is None
    # feature_location='': return raw stage outputs (not incre_modules bottlenecks).
    # index [1] = stride-4, 48ch (raw stage4 branch0) — correct for keypoint detection.
    model = timm.create_model("hrnet_w48", pretrained=use_timm_pretrained,
                              features_only=True, feature_location='')

    if pretrained_path is not None:
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # mmpose checkpoints store weights in 'state_dict' key
        state = ckpt.get("state_dict", ckpt)

        # Strip 'backbone.' prefix to match timm's key naming
        backbone_state = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                new_key = k[len("backbone."):]
                backbone_state[new_key] = v

        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        n_loaded = len(backbone_state) - len(unexpected)
        print(f"HRNet-W48: loaded {n_loaded}/{len(backbone_state)} backbone keys "
              f"from {pretrained_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

    return model


def hrnet_forward(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run HRNet forward pass and return the stride-4 high-resolution feature map.

    With features_only=True and feature_location='', timm's HRNet returns raw stage outputs:
      [0]: stride-2,  64ch,  H/2  x W/2
      [1]: stride-4,  48ch,  H/4  x W/4   <-- we use this one (raw stage4 branch0)
      [2]: stride-8,  96ch,  H/8  x W/8
      [3]: stride-16, 192ch, H/16 x W/16
      [4]: stride-32, 384ch, H/32 x W/32

    For 224px input: index [1] gives (B, 48, 56, 56).

    Args:
        backbone: timm HRNet-W48 model (created with features_only=True, feature_location='')
        x: (B, 3, H, W) input image tensor

    Returns:
        (B, 48, H/4, W/4) stride-4 feature map
    """
    features = backbone(x)  # features_only model uses __call__, not forward_features
    return features[HRNET_FEATURE_IDX]  # (B, 48, H/4, W/4)


def freeze_hrnet(backbone: nn.Module, unfreeze_last_n_stages: int = 0) -> None:
    """Freeze HRNet backbone, optionally keeping last N stages trainable.

    HRNet has 4 stages: stage1 (stem+init), stage2, stage3, stage4.
    Stage4 is the highest-level, multi-resolution fusion stage.

    Args:
        backbone: timm HRNet-W48 model
        unfreeze_last_n_stages: number of final stages to keep trainable (0=fully frozen)
    """
    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    if unfreeze_last_n_stages > 0:
        # timm HRNet stage attribute names (layer1 = initial bottleneck, not stage1)
        stage_names = ["layer1", "stage2", "stage3", "stage4"]
        # Also include transition layers between stages
        transition_names = ["transition1", "transition2", "transition3"]

        stages_to_unfreeze = stage_names[-unfreeze_last_n_stages:]
        transitions_to_unfreeze = transition_names[-(unfreeze_last_n_stages - 1):] if unfreeze_last_n_stages > 1 else []

        for name in stages_to_unfreeze + transitions_to_unfreeze:
            module = getattr(backbone, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

        # When unfreezing all 4 stages, also unfreeze the stem (conv1/bn1/conv2/bn2)
        if unfreeze_last_n_stages >= 4:
            for stem_name in ["conv1", "bn1", "conv2", "bn2"]:
                module = getattr(backbone, stem_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True


def unfreeze_hrnet(backbone: nn.Module) -> None:
    """Unfreeze all HRNet backbone parameters."""
    for param in backbone.parameters():
        param.requires_grad = True


def get_hrnet_norm_modules(backbone: nn.Module) -> list[tuple[str, nn.Module]]:
    """Get all BatchNorm2d modules in the HRNet backbone.

    Used by domain adaptation (TTA, DSU injection) to find normalization layers.

    Returns:
        List of (name, module) tuples for all BatchNorm2d layers.
    """
    return [
        (name, module)
        for name, module in backbone.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    ]


def get_hrnet_stage_modules(backbone: nn.Module, last_n: int) -> list[nn.Module]:
    """Get the last N HRNet stages for DSU/MixStyle hook injection.

    Args:
        backbone: timm HRNet-W48 model
        last_n: number of stages from the end to return

    Returns:
        List of stage nn.Module objects
    """
    stage_names = ["layer1", "stage2", "stage3", "stage4"]
    selected = stage_names[-last_n:] if last_n > 0 else []
    return [getattr(backbone, name) for name in selected if hasattr(backbone, name)]
