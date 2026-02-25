"""Convert colleague's HRNet-W32 checkpoint to our project format.

Loads hrnet_kpts_best.pth, remaps keys to match SatellitePoseModel with
backbone_type="hrnet_w32", and saves in our checkpoint format.

Usage:
    python scripts/convert_colleague_ckpt.py \
        --input /path/to/hrnet_kpts_best.pth \
        --output outputs_hrnet_w32/converted_model.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import SatellitePoseModel


def convert_checkpoint(input_path: str, output_path: str, config_path: str):
    print(f"Loading colleague checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # Extract model state from colleague's format
    src_state = ckpt["model_state"]
    print(f"  Source keys: {len(src_state)}")

    # Remap keys:
    #   backbone.* -> backbone.*  (stays as-is)
    #   head.*     -> keypoint_head.head.*
    #   final.*    -> keypoint_head.final.*
    new_state = {}
    for k, v in src_state.items():
        if k.startswith("backbone."):
            new_state[k] = v
        elif k.startswith("head."):
            new_state[f"keypoint_head.{k}"] = v
        elif k.startswith("final."):
            new_state[f"keypoint_head.{k}"] = v
        else:
            print(f"  WARNING: unmapped key: {k}")

    print(f"  Remapped keys: {len(new_state)}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model and verify strict loading
    pose_cfg = config.get("pose", {})
    geo_cfg = config.get("geometry", {})
    model = SatellitePoseModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=False,
        head_hidden_dims=config["model"]["head_hidden"],
        num_keypoints=config["data"]["num_keypoints"],
        dropout=config["model"]["dropout"],
        mode=config["model"]["mode"],
        keypoint_head_type=config["model"].get("keypoint_head_type", "mlp"),
        heatmap_size=pose_cfg.get("heatmap_size", 128),
        backbone_type=config["model"].get("backbone_type", "hrnet_w32"),
    )

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    # Filter out expected missing keys (grid buffers are registered buffers, not saved in colleague's ckpt)
    real_missing = [k for k in missing if "grid_" not in k]

    print(f"\n  Missing keys: {len(missing)} ({len(real_missing)} non-buffer)")
    if real_missing:
        print(f"    {real_missing[:10]}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"    {unexpected[:10]}")

    if not real_missing and not unexpected:
        print("\n  All model weights matched successfully!")

    # Save in our format
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": ckpt.get("epoch", 0),
        "source": "converted from colleague's hrnet_kpts_best.pth",
    }
    torch.save(save_dict, output_path)
    print(f"\n  Saved converted checkpoint to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert colleague's HRNet-W32 checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to colleague's hrnet_kpts_best.pth")
    parser.add_argument("--output", type=str, default="outputs_hrnet_w32/converted_model.pth",
                        help="Output path for converted checkpoint")
    parser.add_argument("--config", type=str, default="config_hrnet_w32.yaml",
                        help="Config file for the W32 model")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
