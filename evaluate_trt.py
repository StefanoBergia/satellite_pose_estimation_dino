"""Evaluate a TensorRT engine with the same robust EPnP pipeline as evaluate_robust.py.

Supports two modes:
  1. Full evaluation (default): runs evaluate_split → PnP → SLAB score.
  2. FPS benchmark (--benchmark): measures latency/throughput with CUDA events.

The TRTModel wrapper returns {"keypoints": ..., "heatmaps": ...} tensors on CUDA,
matching SatellitePoseModel's output interface so evaluate_split works unchanged.

Usage:
    # Full evaluation (same PnP args as evaluate_robust.py)
    python evaluate_trt.py --engine hrnet_w32.engine --config config_hrnet_w32.yaml --gt_crop --crop_pnp --resize_first --kpt_extractor argmax --refine_lm 1 --refine_retrim 1 --min_inliers_schedule "11,9,8,6,4" --no_conf_filter

    # FPS benchmark
    python evaluate_trt.py --engine hrnet_w32.engine --config config_hrnet_w32.yaml --benchmark --batch_sizes 1 4 8
"""

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_robust import (
    evaluate_split,
    compute_pose_errors,
    print_selection_report,
    print_results_table,
    save_results,
)
from src.dataset import SpeedPlusKeypointDataset
from src.transforms import KeypointTransform
from src.utils import load_pnp_data


# ---------------------------------------------------------------------------
# TensorRT model wrapper
# ---------------------------------------------------------------------------

class TRTModel:
    """Wraps a TensorRT engine to match SatellitePoseModel's dict-output interface.

    Uses the TensorRT 10.x Python API:
      - engine.get_tensor_name / get_tensor_mode / get_tensor_shape
      - context.set_input_shape / set_tensor_address / execute_async_v3
    """

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Deserialize engine
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover tensor names and their roles
        self.input_names = []
        self.output_names = []
        self.output_shapes = {}  # name -> tuple (without batch dim resolved)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                # Shape may have -1 for dynamic batch dim
                self.output_shapes[name] = tuple(self.engine.get_tensor_shape(name))

        # Pre-allocate output buffers (will be resized if batch changes)
        self._output_buffers = {}
        self._last_batch_size = 0

        # CUDA stream for async execution
        self.stream = torch.cuda.Stream(device=self.device)

    def _ensure_buffers(self, batch_size):
        """Allocate output buffers for the given batch size if needed."""
        if batch_size == self._last_batch_size:
            return
        self._last_batch_size = batch_size

        for name in self.output_names:
            shape = list(self.output_shapes[name])
            # Replace dynamic dim (-1) with actual batch size
            shape = [batch_size if s == -1 else s for s in shape]
            self._output_buffers[name] = torch.zeros(
                shape, dtype=torch.float32, device=self.device
            )

    def __call__(self, pixel_values, **kwargs):
        """Run inference on a batch of images.

        Args:
            pixel_values: (B, 3, H, W) float32 tensor on CUDA.

        Returns:
            dict with "keypoints" (B, K, 2) and "heatmaps" (B, K, Hm, Wm).
        """
        B = pixel_values.shape[0]
        pixel_values = pixel_values.contiguous().to(self.device)

        # Set dynamic input shape and allocate outputs
        input_name = self.input_names[0]
        ok = self.context.set_input_shape(input_name, tuple(pixel_values.shape))
        if not ok:
            raise RuntimeError(
                f"TRT rejected input shape {tuple(pixel_values.shape)}. "
                f"Engine max dynamic batch may be smaller than batch_size={B}."
            )
        self._ensure_buffers(B)

        # Bind tensor addresses
        self.context.set_tensor_address(input_name, pixel_values.data_ptr())
        for name in self.output_names:
            self.context.set_tensor_address(name, self._output_buffers[name].data_ptr())

        # Execute
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Build output dict matching SatellitePoseModel's convention
        result = {}
        for name in self.output_names:
            buf = self._output_buffers[name]
            key = name.lower()
            result[key] = buf.clone()

        # One-time diagnostic: print output stats on first call
        if not hasattr(self, "_diag_done"):
            self._diag_done = True
            for key, val in result.items():
                n_nan = torch.isnan(val).sum().item()
                n_inf = torch.isinf(val).sum().item()
                finite = val[torch.isfinite(val)]
                if finite.numel() > 0:
                    print(f"  [DIAG] {key}: shape={list(val.shape)}, "
                          f"min={finite.min():.4f}, max={finite.max():.4f}, "
                          f"mean={finite.mean():.4f}, NaN={n_nan}, Inf={n_inf}")
                else:
                    print(f"  [DIAG] {key}: shape={list(val.shape)}, "
                          f"ALL NaN/Inf ({n_nan} NaN, {n_inf} Inf)")

        return result

    def eval(self):
        """No-op: TRT engine is always in inference mode."""
        return self


# ---------------------------------------------------------------------------
# FPS benchmark
# ---------------------------------------------------------------------------

def benchmark_fps(engine_path, config, batch_sizes, warmup_iters, bench_iters):
    """Measure inference latency and throughput for a TRT engine.

    Uses the same methodology as benchmark_fps.py: CUDA events for timing,
    OOM-safe iteration, median/mean/std reporting.
    """
    image_size = config["data"]["image_size"]
    device = torch.device("cuda")

    model = TRTModel(engine_path, device="cuda")
    print(f"\nEngine: {engine_path}")
    print(f"Input:  (B, 3, {image_size}, {image_size})")
    print(f"Warmup: {warmup_iters} iters | Bench: {bench_iters} iters")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n  {'Batch':>5s} | {'FPS':>8s} | {'Latency (ms)':^22s} | {'Peak Mem (MB)':>13s}")
    print(f"  {'-'*5:>5s}-+-{'-'*8:>8s}-+-{'-'*22:^22s}-+-{'-'*13:>13s}")

    for bs in batch_sizes:
        dummy = torch.randn(bs, 3, image_size, image_size, device=device)

        # Warmup (OOM-safe)
        try:
            for _ in range(warmup_iters):
                model(dummy)
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:5d} | OOM during warmup, skipping remaining batch sizes")
            break

        torch.cuda.reset_peak_memory_stats(device)

        # Timed iterations (OOM-safe)
        latencies = []
        try:
            for _ in range(bench_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(dummy)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:5d} | OOM during benchmark, skipping remaining batch sizes")
            break

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        fps = bs / (mean_lat / 1000.0)

        print(f"  {bs:5d} | {fps:8.1f} | "
              f"{mean_lat:7.2f} +/- {std_lat:5.2f}      | {peak_mem:13.0f}")

        del dummy
        torch.cuda.empty_cache()

    print()


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args, config):
    """Run the same robust EPnP evaluation pipeline as evaluate_robust.py."""
    import datetime

    device = torch.device("cuda")

    mode = config["model"]["mode"]
    root = Path(config["data"]["root"])
    geo_cfg = config.get("geometry", {})
    pose_cfg = config.get("pose", {})

    # Print engine file info for debugging
    engine_path = Path(args.engine)
    if engine_path.exists():
        mtime = datetime.datetime.fromtimestamp(engine_path.stat().st_mtime)
        fsize = engine_path.stat().st_size / (1024 ** 2)
        print(f"Engine:  {args.engine} ({fsize:.1f} MB, modified {mtime})")
    else:
        print(f"Engine:  {args.engine}")
    print(f"Config:  {args.config}")
    print(f"Mode:    {mode}")
    print(f"Device:  {device}")

    # Print PnP settings
    print(f"\n  PnP settings:")
    print(f"    Confidence threshold:         {args.confidence}")
    print(f"    Minimum landmarks:            {args.min_landmarks}")
    print(f"    RANSAC reprojection error:    {args.reproj_error} px")
    print(f"    RANSAC confidence:            {args.ransac_confidence}")
    print(f"    RANSAC iterations:            {args.ransac_iterations}")
    print(f"    Cascading inlier schedule:    {args.min_inliers_schedule_list}")
    print(f"    LM refinement:               {'ON' if args.refine_lm else 'OFF'}")
    print(f"    LM retrim:                   {'ON' if args.refine_retrim else 'OFF'} "
          f"(keep_frac={args.refine_keep_frac}, min_keep={args.refine_min_keep})")
    print(f"    GT crop:                     {'ON' if args.gt_crop else 'OFF'}")
    print(f"    Resize first:                {'ON' if args.resize_first else 'OFF'}")
    print(f"    Crop-space PnP:              {'ON' if args.crop_pnp else 'OFF'}")
    print(f"    Keypoint extractor:          {args.kpt_extractor}")
    print(f"    Confidence pre-filter:       {'OFF (all visible → RANSAC)' if args.no_conf_filter else 'ON'}")
    if args.rmse_inliers_thr > 0:
        print(f"    Inlier RMSE gate:            {args.rmse_inliers_thr} px")
    if args.min_kpt_area > 0:
        print(f"    Min keypoint area gate:      {args.min_kpt_area} px²")
    if args.t_ratio_max > 0:
        print(f"    Translation ratio gate:      {args.t_ratio_max}x (GT-free, crop-based z_expected)")
    print(f"    Fallback: if < {args.min_landmarks} keypoints pass threshold, "
          f"use top-{args.min_landmarks} by confidence")

    settings = {
        "confidence": args.confidence,
        "min_landmarks": args.min_landmarks,
        "reproj_error": args.reproj_error,
    }

    # Load PnP data
    if not (geo_cfg.get("points_3d") and geo_cfg.get("camera")):
        print("ERROR: No geometry config found. Cannot run PnP.")
        return
    pnp_data = load_pnp_data(geo_cfg["points_3d"], geo_cfg["camera"])
    print(f"  3D points: {pnp_data['points_3d'].shape}")

    model_extent = float(np.ptp(pnp_data["points_3d"], axis=0).max())

    # Build TRT model
    model = TRTModel(args.engine, device="cuda")

    transform = KeypointTransform(
        image_size=config["data"]["image_size"],
        is_train=False,
        imagenet_normalize=config["data"].get("imagenet_normalize", True),
    )
    occluded_weight = config["train"].get("occluded_weight", 0.5)
    pck_threshold = config.get("eval", {}).get("pck_threshold", 0.05)

    all_results = {}
    for split in args.splits:
        if split not in config["data"]["splits"]:
            print(f"  Skipping {split} (not in config)")
            continue

        split_cfg = config["data"]["splits"][split]

        pose_json = None
        pose_labels = config["data"].get("pose_labels", {})
        if split in pose_labels:
            pose_json = pose_labels[split]

        include_list = None
        if split in ("lightbox", "sunlamp"):
            if args.test_split:
                test_list_path = Path(args.splits_dir) / f"{split}_test.txt"
                if test_list_path.exists():
                    with open(test_list_path) as f:
                        include_list = set(line.strip() for line in f if line.strip())
                    print(f"  Using test split only: {len(include_list)} images "
                          f"(excluded style subset)")
                else:
                    print(f"  WARNING: {test_list_path} not found, using full split")

        dataset = SpeedPlusKeypointDataset(
            image_dir=str(root / split_cfg["images"]),
            label_dir=str(root / split_cfg["labels"]),
            num_keypoints=config["data"]["num_keypoints"],
            bbox_pad_ratio=config["data"].get("bbox_pad_ratio", 0.1),
            transform=transform,
            pose_json=pose_json,
            include_list=include_list,
            no_crop=args.no_crop,
            gt_crop=args.gt_crop,
            resize_first=config["data"]["image_size"] if args.resize_first else 0,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"\n{'':=<80}")
        print(f"  Evaluating: {split} ({len(dataset)} samples)")
        print(f"{'':=<80}")

        kp_metrics, pnp_results, pose_errors, method_counts, _, _ = evaluate_split(
            model, loader, mode, device, pnp_data,
            min_landmarks=args.min_landmarks,
            reproj_error=args.reproj_error,
            confidence_threshold=args.confidence,
            occluded_weight=occluded_weight,
            pck_threshold=pck_threshold,
            min_inliers_schedule=args.min_inliers_schedule_list,
            refine_lm=bool(args.refine_lm),
            refine_retrim=bool(args.refine_retrim),
            refine_keep_frac=args.refine_keep_frac,
            refine_min_keep=args.refine_min_keep,
            crop_pnp=args.crop_pnp,
            image_size=config["data"]["image_size"],
            ransac_iterations=args.ransac_iterations,
            ransac_confidence=args.ransac_confidence,
            min_kpt_area=args.min_kpt_area,
            t_ratio_max=args.t_ratio_max,
            model_extent=model_extent,
            resize_first=args.resize_first,
            heatmap_size=pose_cfg.get("heatmap_size", 128),
            kpt_extractor=args.kpt_extractor,
            rmse_inliers_thr=args.rmse_inliers_thr,
            no_conf_filter=args.no_conf_filter,
        )

        n_total = len(pnp_results)
        n_solved = sum(1 for r in pnp_results if r["success"])
        n_dropped = n_total - n_solved

        print_selection_report(method_counts, pnp_results, args.confidence,
                               args.min_landmarks, n_total)

        solved_errors = [e for e in pose_errors if e is not None]
        if solved_errors:
            mean_slab = np.mean([e["slab"] for e in solved_errors])
            mean_ori = np.mean([e["orient_score"] for e in solved_errors])
            mean_pos = np.mean([e["pos_score"] for e in solved_errors])
            mean_rot = np.mean([e["rot_deg"] for e in solved_errors])
            mean_t = np.mean([e["pos_abs"] for e in solved_errors])
        else:
            mean_slab = mean_ori = mean_pos = mean_rot = mean_t = 0.0

        inliers = [r["n_inliers"] for r in pnp_results if r["success"]]
        if inliers:
            print(f"  Inlier stats: min={min(inliers)}, "
                  f"median={np.median(inliers):.0f}, "
                  f"mean={np.mean(inliers):.1f}, "
                  f"max={max(inliers)}")

        n_kp_used = [r["n_keypoints_used"] for r in pnp_results if r["success"]]
        if n_kp_used:
            print(f"  Keypoints used: min={min(n_kp_used)}, "
                  f"median={np.median(n_kp_used):.0f}, "
                  f"mean={np.mean(n_kp_used):.1f}, "
                  f"max={max(n_kp_used)}")

        all_results[split] = {
            **kp_metrics,
            "epnp_slab": mean_slab,
            "epnp_ori": mean_ori,
            "epnp_pos": mean_pos,
            "epnp_rot": mean_rot,
            "epnp_t": mean_t,
            "solved%": n_solved / max(n_total, 1),
            "dropped": n_dropped,
        }

    print_results_table(all_results, mode, "TRT", settings)

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.engine).parent / "results_trt.txt")
    save_results(all_results, output_path, mode, "TRT", settings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a TensorRT engine with robust EPnP + FPS benchmark"
    )
    parser.add_argument("--engine", type=str, required=True, help="Path to .engine file")
    parser.add_argument("--config", type=str, required=True, help="YAML config path (model/data settings)")

    # Mode
    parser.add_argument("--benchmark", action="store_true", help="Run FPS benchmark instead of evaluation")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Batch sizes for benchmark (default: 1 2 4 8 16)")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Warmup iterations for benchmark (default: 50)")
    parser.add_argument("--bench_iters", type=int, default=200, help="Timed iterations for benchmark (default: 200)")

    # Evaluation args (mirrors evaluate_robust.py)
    parser.add_argument("--splits", type=str, nargs="+", default=["val", "lightbox", "sunlamp"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    # Robust PnP parameters
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min_landmarks", type=int, default=8)
    parser.add_argument("--reproj_error", type=float, default=15.0)
    parser.add_argument("--no_crop", action="store_true", help="Feed full images (no YOLO bbox crop)")
    parser.add_argument("--output", type=str, default=None, help="Output results file (default: <engine_dir>/results_trt.txt)")
    parser.add_argument("--test_split", action="store_true", help="Evaluate only test subset for lightbox/sunlamp")
    parser.add_argument("--splits_dir", type=str, default="data/splits")

    # Cascading inlier schedule
    parser.add_argument("--min_inliers_schedule", type=str, default="", help="Comma-separated descending min inlier thresholds (e.g. '11,9,8,6,4')")

    # LM refinement
    parser.add_argument("--refine_lm", type=int, default=0)
    parser.add_argument("--refine_retrim", type=int, default=0)
    parser.add_argument("--refine_keep_frac", type=float, default=0.8)
    parser.add_argument("--refine_min_keep", type=int, default=6)

    # GT crop + crop-space PnP
    parser.add_argument("--gt_crop", action="store_true")
    parser.add_argument("--crop_pnp", action="store_true")
    parser.add_argument("--ransac_confidence", type=float, default=0.99)
    parser.add_argument("--ransac_iterations", type=int, default=200)
    parser.add_argument("--min_kpt_area", type=float, default=0.0)
    parser.add_argument("--t_ratio_max", type=float, default=0.0)
    parser.add_argument("--resize_first", action="store_true")
    parser.add_argument("--kpt_extractor", type=str, default="softargmax", choices=["softargmax", "argmax"])
    parser.add_argument("--rmse_inliers_thr", type=float, default=0.0)
    parser.add_argument("--no_conf_filter", action="store_true")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.benchmark:
        benchmark_fps(args.engine, config, args.batch_sizes,
                      args.warmup_iters, args.bench_iters)
    else:
        # Parse cascading schedule
        schedule_str = (args.min_inliers_schedule or "").strip()
        if schedule_str:
            args.min_inliers_schedule_list = sorted(
                [int(x.strip()) for x in schedule_str.split(",") if x.strip()],
                reverse=True,
            )
        else:
            args.min_inliers_schedule_list = None

        run_evaluation(args, config)


if __name__ == "__main__":
    main()
