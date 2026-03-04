"""
Export DINO ViT-L/16 satellite pose estimation model to ONNX (and optionally TensorRT).

Exports the DINO heatmap-based keypoint model for deployment on NVIDIA Jetson Orin.
The ONNX model is the primary deliverable; TensorRT engine should be rebuilt on-device.

Usage:
    python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth
    python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth --validate
    python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth --build_trt --fp16
    python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth --build_trt --int8 --calib_images images/val --calib_labels labels/val
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image

from src.model import SatellitePoseModel


class OnnxExportWrapper(torch.nn.Module):
    """Wrapper that converts dict outputs to a tuple for ONNX export."""

    def __init__(self, model: SatellitePoseModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values)
        return out["keypoints"], out["heatmaps"]


def build_model(config, checkpoint_path, device):
    """Build and load DINO model in keypoint_only mode."""
    pose_cfg = config.get("pose", {})

    model = SatellitePoseModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=True,
        head_hidden_dims=config["model"]["head_hidden"],
        num_keypoints=config["data"]["num_keypoints"],
        dropout=config["model"]["dropout"],
        mode="keypoint_only",
        keypoint_head_type=config["model"].get("keypoint_head_type", "mlp"),
        heatmap_size=pose_cfg.get("heatmap_size", 64),
        backbone_type=config["model"].get("backbone_type", "dinov3"),
        hrnet_pretrained=config["model"].get("hrnet_pretrained", None),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in model.backbone.parameters()) if hasattr(model, "backbone") else 0
    head = sum(p.numel() for p in model.keypoint_head.parameters()) if hasattr(model, "keypoint_head") else 0
    return total, backbone, head


def export_onnx(wrapper, image_size, output_path, dynamic_batch, opset=17):
    """Export the wrapped model to ONNX."""
    device = next(wrapper.parameters()).device
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "keypoints": {0: "batch_size"},
            "heatmaps": {0: "batch_size"},
        }

    print(f"Exporting ONNX (opset {opset}, dynamic_batch={dynamic_batch})...")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["keypoints", "heatmaps"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    elapsed = time.time() - t0

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"ONNX exported to {output_path} ({elapsed:.1f}s, {file_size_mb:.1f} MB)")


def validate_onnx(wrapper, onnx_path, image_size, device):
    """Validate ONNX model outputs against PyTorch outputs."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("WARNING: onnx/onnxruntime not installed, skipping validation")
        return False

    # Check ONNX model is well-formed
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model passed checker validation")

    # Print input/output tensor info
    graph = model_onnx.graph
    print("Input tensors:")
    for inp in graph.input:
        print(f"  {inp.name}: {[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}")
    print("Output tensors:")
    for out in graph.output:
        print(f"  {out.name}: {[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}")

    # Compare outputs
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        pt_kp, pt_hm = wrapper(dummy_input)
    pt_kp = pt_kp.cpu().numpy()
    pt_hm = pt_hm.cpu().numpy()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {"pixel_values": dummy_input.cpu().numpy()}
    ort_kp, ort_hm = session.run(None, ort_inputs)

    kp_diff = np.abs(pt_kp - ort_kp).max()
    hm_diff = np.abs(pt_hm - ort_hm).max()

    print(f"Validation: keypoints max_diff={kp_diff:.6e}, heatmaps max_diff={hm_diff:.6e}")

    tol = 1e-4
    if kp_diff < tol and hm_diff < tol:
        print(f"PASS: all outputs within tolerance ({tol})")
        return True
    else:
        print(f"WARNING: outputs exceed tolerance ({tol})")
        return False


def crop_bbox(image, bbox, bbox_pad_ratio=0.1):
    """Crop image to YOLO bbox with padding (matches dataset._crop_bbox).

    Args:
        image: PIL Image.
        bbox: (cx, cy, w, h) normalized [0,1].
        bbox_pad_ratio: Padding as fraction of bbox size.

    Returns:
        Cropped PIL Image.
    """
    img_w, img_h = image.size
    cx, cy, w, h = bbox
    cx_px, cy_px = cx * img_w, cy * img_h
    w_px, h_px = w * img_w, h * img_h
    pad_w, pad_h = w_px * bbox_pad_ratio, h_px * bbox_pad_ratio

    x1 = max(0, int(cx_px - w_px / 2 - pad_w))
    y1 = max(0, int(cy_px - h_px / 2 - pad_h))
    x2 = min(img_w, int(cx_px + w_px / 2 + pad_w))
    y2 = min(img_h, int(cy_px + h_px / 2 + pad_h))

    return image.crop((x1, y1, x2, y2))


def parse_yolo_bbox(label_path):
    """Parse bbox (cx, cy, w, h) from a YOLO keypoint label file."""
    with open(label_path, "r") as f:
        tokens = f.readline().strip().split()
    return [float(t) for t in tokens[1:5]]


def load_calibration_images(image_dir, label_dir, image_size, num_images,
                            bbox_pad_ratio=0.1, imagenet_normalize=False):
    """Load bbox-cropped calibration images as numpy arrays in NCHW format.

    Replicates the inference preprocessing: load image, crop to YOLO bbox
    with padding, resize to target size, normalize.

    Args:
        image_dir: Directory containing .jpg images.
        label_dir: Directory containing YOLO .txt labels (same stems as images).
        image_size: Target image size (square).
        num_images: Max number of images to load.
        bbox_pad_ratio: Padding around bbox as fraction of bbox size.
        imagenet_normalize: Whether to apply ImageNet normalization.

    Returns:
        List of numpy arrays, each shape (1, 3, H, W), float32.
    """
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")

    # Filter to images that have a matching label
    paired = []
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        lbl = os.path.join(label_dir, stem + ".txt")
        if os.path.isfile(lbl):
            paired.append((p, lbl))
    if not paired:
        raise FileNotFoundError(f"No matching .txt labels in {label_dir} for images in {image_dir}")
    paired = paired[:num_images]
    print(f"Loading {len(paired)} calibration images from {image_dir} (bbox crop + resize to {image_size}x{image_size})")

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    images = []
    for img_path, lbl_path in paired:
        img = Image.open(img_path).convert("RGB")
        bbox = parse_yolo_bbox(lbl_path)
        crop = crop_bbox(img, bbox, bbox_pad_ratio)
        crop = crop.resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(crop, dtype=np.float32) / 255.0  # HWC [0,1]
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]    # NCHW
        if imagenet_normalize:
            arr = (arr - imagenet_mean) / imagenet_std
        images.append(arr)

    return images


def build_trt_engine(onnx_path, trt_path, image_size, fp16, int8, dynamic_batch,
                     calib_images=None, calib_labels=None, calib_cache=None,
                     num_calib_images=200, bbox_pad_ratio=0.1,
                     imagenet_normalize=False, force_all_fp32=False):
    """Build TensorRT engine using the Python API with optional INT8 calibration."""
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt Python package not installed. Install it or use trtexec manually.")
        return False

    if not torch.cuda.is_available():
        print("ERROR: TensorRT engine build requires a CUDA GPU. Run this on a GPU node (e.g. via SLURM).")
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # --- INT8 Calibrator ---
    class INT8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, image_dir, label_dir, img_size, num_images, cache_file,
                     pad_ratio, use_imagenet_norm):
            super().__init__()
            self.cache_file = cache_file
            self.images = load_calibration_images(
                image_dir, label_dir, img_size, num_images,
                bbox_pad_ratio=pad_ratio, imagenet_normalize=use_imagenet_norm,
            )
            self.batch_idx = 0

            # Use PyTorch CUDA for device memory (avoids pycuda dependency)
            self.device_tensor = torch.zeros(
                self.images[0].shape, dtype=torch.float32, device="cuda"
            )

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.batch_idx >= len(self.images):
                return None
            img = torch.from_numpy(np.ascontiguousarray(self.images[self.batch_idx]))
            self.device_tensor.copy_(img)
            self.batch_idx += 1
            return [self.device_tensor.data_ptr()]

        def read_calibration_cache(self):
            if self.cache_file and os.path.isfile(self.cache_file):
                print(f"Reading INT8 calibration cache from {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_file:
                print(f"Writing INT8 calibration cache to {self.cache_file}")
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

    # --- Build engine ---
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX model: {onnx_path}")
    if not parser.parse_from_file(os.path.abspath(onnx_path)):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}")
        return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    # Optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    if dynamic_batch:
        min_b, opt_b, max_b = dynamic_batch
    else:
        min_b = opt_b = max_b = 1
    profile.set_shape("pixel_values",
                      (min_b, 3, image_size, image_size),
                      (opt_b, 3, image_size, image_size),
                      (max_b, 3, image_size, image_size))
    config.add_optimization_profile(profile)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")

            # Mixed precision: force numerically sensitive ViT layers to FP32.
            # ViT attention logits, LayerNorm reductions, and soft-argmax
            # can overflow FP16 max (~65504). Force these layer TYPES to FP32
            # rather than name-matching (TRT layer names differ from ONNX names).
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

            if force_all_fp32:
                # Debug mode: force ALL layers to FP32 (keeps FP16 weight storage)
                fp32_types = None  # match everything
                print("FORCE_ALL_FP32: every layer will run in FP32")
            else:
                fp32_types = {
                    trt.LayerType.SOFTMAX,          # exp() overflow in attention + soft-argmax
                    trt.LayerType.MATRIX_MULTIPLY,  # QK^T overflow, accumulation in projections
                    trt.LayerType.NORMALIZATION,     # fused LayerNorm: sqrt/div overflow
                    trt.LayerType.REDUCE,            # mean/var if LayerNorm is not fused
                }

            # Integer types that cannot be forced to FP32
            int_dtypes = {trt.int32, trt.int64, trt.bool}

            n_forced = 0
            n_skipped_int = 0
            type_counts = {}
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                type_name = str(layer.type).split(".")[-1]
                type_counts[type_name] = type_counts.get(type_name, 0) + 1

                if fp32_types is None or layer.type in fp32_types:
                    # Skip layers with integer outputs (shape ops, constants)
                    has_int_output = any(
                        layer.get_output(j).dtype in int_dtypes
                        for j in range(layer.num_outputs)
                    )
                    if has_int_output:
                        n_skipped_int += 1
                        continue
                    layer.precision = trt.float32
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.float32)
                    n_forced += 1

            total_layers = network.num_layers
            print(f"Mixed precision: {n_forced}/{total_layers} layers forced to FP32"
                  f" ({n_skipped_int} int-typed layers skipped)")
            print("  Layer type distribution:")
            for t, c in sorted(type_counts.items()):
                print(f"    {t}: {c}")
        else:
            print("WARNING: Platform does not support fast FP16, ignoring --fp16")

    calibrator = None
    if int8:
        if not builder.platform_has_fast_int8:
            print("WARNING: Platform does not support fast INT8, proceeding anyway")
        config.set_flag(trt.BuilderFlag.INT8)
        print("NOTE: INT8 quantization for ViT models may degrade accuracy — verify with --validate")
        if calib_images and calib_labels:
            cache_path = calib_cache or "dino_int8_calib.cache"
            calibrator = INT8Calibrator(calib_images, calib_labels, image_size,
                                        num_calib_images, cache_path,
                                        bbox_pad_ratio, imagenet_normalize)
            config.int8_calibrator = calibrator
            print(f"INT8 calibration enabled ({num_calib_images} images)")
        else:
            print("WARNING: INT8 without --calib_images/--calib_labels, TensorRT will use random calibration")

    print("Building TensorRT engine (this may take several minutes)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if serialized_engine is None:
        print("ERROR: TensorRT engine build failed")
        return False

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)

    file_size_mb = os.path.getsize(trt_path) / (1024 ** 2)
    print(f"TensorRT engine saved to {trt_path} ({elapsed:.1f}s, {file_size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export DINO ViT-L/16 model to ONNX/TensorRT")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_onnx", type=str, default="model_dino.onnx", help="Output ONNX path")
    parser.add_argument("--output_trt", type=str, default="model_dino.engine", help="Output TensorRT engine path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--validate", action="store_true", help="Validate ONNX against PyTorch")
    parser.add_argument("--build_trt", action="store_true", help="Build TensorRT engine")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 for TensorRT")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 for TensorRT (requires --calib_images)")
    parser.add_argument("--dynamic_batch", type=int, nargs=3, metavar=("MIN", "OPT", "MAX"), default=None, help="Dynamic batch sizes: min opt max")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export (cpu recommended for portability)")
    parser.add_argument("--calib_images", type=str, default=None, help="Directory with .jpg images for INT8 calibration")
    parser.add_argument("--calib_labels", type=str, default=None, help="Directory with YOLO .txt labels for bbox cropping during calibration")
    parser.add_argument("--calib_cache", type=str, default=None, help="Path to save/load INT8 calibration cache")
    parser.add_argument("--num_calib_images", type=int, default=200, help="Number of calibration images for INT8")
    parser.add_argument("--force_all_fp32", action="store_true", help="Force ALL layers to FP32 (debug: isolates FP16 precision issues)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    image_size = config["data"]["image_size"]
    num_keypoints = config["data"]["num_keypoints"]
    heatmap_size = config.get("pose", {}).get("heatmap_size", 64)
    imagenet_normalize = config.get("training", {}).get("imagenet_normalize", True)

    device = torch.device(args.device)

    # Build model
    print(f"Loading model from {args.checkpoint}...")
    model = build_model(config, args.checkpoint, device)
    total, backbone, head = count_parameters(model)
    print(f"Parameters: {total/1e6:.2f}M total ({backbone/1e6:.2f}M backbone, {head/1e6:.2f}M head)")
    print(f"Input:  (B, 3, {image_size}, {image_size})")
    print(f"Output: keypoints (B, {num_keypoints}, 2), heatmaps (B, {num_keypoints}, {heatmap_size}, {heatmap_size})")

    # Monkey-patch heatmap head for TRT-compatible ONNX export:
    # The deconv stack outputs 56x56 natively (14->28->56). When heatmap_size=64,
    # the forward() uses F.interpolate to upsample 56->64, but TRT miscomputes
    # the Resize op output shape causing a downstream Reshape volume mismatch.
    # Fix: set heatmap_size=56 so the F.interpolate branch is never taken.
    head = model.keypoint_head
    if hasattr(head, "heatmap_size") and head.heatmap_size != 56:
        native_size = 56  # 14 -> 28 -> 56 via two deconv layers
        print(f"TRT export: patching heatmap_size {head.heatmap_size} -> {native_size} (removes F.interpolate from graph)")
        head.heatmap_size = native_size
        # Rebuild soft-argmax coordinate grids at the native size
        x_coords = torch.linspace(0, 1, native_size)
        y_coords = torch.linspace(0, 1, native_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        head.register_buffer("grid_x", xx.reshape(1, 1, -1).to(device))
        head.register_buffer("grid_y", yy.reshape(1, 1, -1).to(device))
        heatmap_size = native_size  # update for print below
        print(f"Output: keypoints (B, {num_keypoints}, 2), heatmaps (B, {num_keypoints}, {native_size}, {native_size})")

    # Wrap for ONNX export
    wrapper = OnnxExportWrapper(model)
    wrapper.eval()

    # Export ONNX
    enable_dynamic = args.dynamic_batch is not None
    export_onnx(wrapper, image_size, args.output_onnx, dynamic_batch=enable_dynamic, opset=args.opset)

    # Validate
    if args.validate:
        validate_onnx(wrapper, args.output_onnx, image_size, device)

    # Build TensorRT engine
    if args.build_trt:
        if args.dynamic_batch is None:
            print("NOTE: --dynamic_batch not set, TRT engine will use static batch=1")
        build_trt_engine(
            args.output_onnx, args.output_trt, image_size,
            args.fp16, args.int8, args.dynamic_batch,
            calib_images=args.calib_images,
            calib_labels=args.calib_labels,
            calib_cache=args.calib_cache,
            num_calib_images=args.num_calib_images,
            bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
            imagenet_normalize=imagenet_normalize,
            force_all_fp32=args.force_all_fp32,
        )


if __name__ == "__main__":
    main()
