"""
Benchmark inference FPS for DINO, HRNet-W32, and TensorRT models.

Usage:
    python benchmark_fps.py --config_a config.yaml --config_b config_hrnet_w32.yaml
    python benchmark_fps.py --config_a config.yaml --only_a --amp
    python benchmark_fps.py --config_a config.yaml --checkpoint_a outputs_keypoints_heatmap/best_model.pth --config_b config_hrnet_w32.yaml --checkpoint_b outputs_hrnet_w32/converted_model.pth
    python benchmark_fps.py --trt_engines hrnet_w32.engine hrnet_w32_fp16.engine --trt_config config_hrnet_w32.yaml
    python benchmark_fps.py --config_a config.yaml --only_a --trt_engines hrnet_w32.engine --trt_config config_hrnet_w32.yaml
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import yaml

from src.model import SatellitePoseModel


# ---------------------------------------------------------------------------
# TensorRT model wrapper (self-contained, no dependency on evaluate_trt.py)
# ---------------------------------------------------------------------------

class TRTModel:
    """Wraps a TensorRT engine for benchmarking. Returns dict output like SatellitePoseModel."""

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)

        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        self.output_shapes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(self.engine.get_tensor_shape(name))

        self._output_buffers = {}
        self._last_batch_size = 0
        self.stream = torch.cuda.Stream(device=self.device)

    def _ensure_buffers(self, batch_size):
        if batch_size == self._last_batch_size:
            return
        self._last_batch_size = batch_size
        for name in self.output_names:
            shape = [batch_size if s == -1 else s for s in self.output_shapes[name]]
            self._output_buffers[name] = torch.zeros(
                shape, dtype=torch.float32, device=self.device
            )

    def __call__(self, pixel_values, **kwargs):
        B = pixel_values.shape[0]
        pixel_values = pixel_values.contiguous().to(self.device)

        input_name = self.input_names[0]
        self.context.set_input_shape(input_name, tuple(pixel_values.shape))
        self._ensure_buffers(B)

        self.context.set_tensor_address(input_name, pixel_values.data_ptr())
        for name in self.output_names:
            self.context.set_tensor_address(name, self._output_buffers[name].data_ptr())

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return {name.lower(): self._output_buffers[name].clone() for name in self.output_names}

    def eval(self):
        return self


def build_model(config, checkpoint_path=None, device="cuda"):
    """Build a SatellitePoseModel from config, optionally loading weights."""
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

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    """Count total, backbone, and head parameters."""
    total = sum(p.numel() for p in model.parameters())
    backbone_params = 0
    head_params = 0

    if hasattr(model, "backbone"):
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
    if hasattr(model, "hrnet"):
        backbone_params = sum(p.numel() for p in model.hrnet.parameters())

    if hasattr(model, "keypoint_head"):
        head_params = sum(p.numel() for p in model.keypoint_head.parameters())

    return {"total": total, "backbone": backbone_params, "head": head_params}


@torch.no_grad()
def benchmark_model(model, input_shape, batch_sizes, warmup_iters, bench_iters, use_amp=False):
    """Benchmark a model across multiple batch sizes. Returns list of result dicts."""
    device = next(model.parameters()).device
    results = []

    for bs in batch_sizes:
        x = torch.randn(bs, *input_shape, device=device)
        torch.cuda.reset_peak_memory_stats(device)

        # Warmup
        try:
            for _ in range(warmup_iters):
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        model(x)
                else:
                    model(x)
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {bs:>4d}: OOM during warmup, skipping remaining batch sizes")
            break

        # Timed iterations
        latencies = []
        try:
            for _ in range(bench_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        model(x)
                else:
                    model(x)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {bs:>4d}: OOM during benchmark, skipping remaining batch sizes")
            break

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        min_lat = min(latencies)
        max_lat = max(latencies)
        med_lat = statistics.median(latencies)
        fps = bs / (mean_lat / 1000.0)

        results.append({
            "batch_size": bs,
            "mean_ms": mean_lat,
            "std_ms": std_lat,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "median_ms": med_lat,
            "fps": fps,
            "throughput_img_s": fps,
            "peak_mem_mb": peak_mem,
            "num_iters": len(latencies),
        })

        del x
        torch.cuda.empty_cache()

    return results


@torch.no_grad()
def benchmark_trt(engine_path, image_size, batch_sizes, warmup_iters, bench_iters):
    """Benchmark a TensorRT engine across multiple batch sizes. Returns list of result dicts."""
    device = torch.device("cuda")
    model = TRTModel(engine_path, device="cuda")
    results = []

    for bs in batch_sizes:
        x = torch.randn(bs, 3, image_size, image_size, device=device)
        torch.cuda.reset_peak_memory_stats(device)

        # Warmup
        try:
            for _ in range(warmup_iters):
                model(x)
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {bs:>4d}: OOM during warmup, skipping remaining batch sizes")
            break

        # Timed iterations
        latencies = []
        try:
            for _ in range(bench_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {bs:>4d}: OOM during benchmark, skipping remaining batch sizes")
            break

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        min_lat = min(latencies)
        max_lat = max(latencies)
        med_lat = statistics.median(latencies)
        fps = bs / (mean_lat / 1000.0)

        results.append({
            "batch_size": bs,
            "mean_ms": mean_lat,
            "std_ms": std_lat,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "median_ms": med_lat,
            "fps": fps,
            "throughput_img_s": fps,
            "peak_mem_mb": peak_mem,
            "num_iters": len(latencies),
        })

        del x
        torch.cuda.empty_cache()

    return results


def print_results(name, image_size, params, results, use_amp=False):
    """Print formatted benchmark results."""
    amp_tag = " [AMP FP16]" if use_amp else ""
    if params:
        print(f"\n{name}  |  {image_size}x{image_size}  |  {params['total']/1e6:.1f}M params "
              f"(backbone {params['backbone']/1e6:.1f}M, head {params['head']/1e6:.1f}M){amp_tag}")
    else:
        engine_size = ""
        print(f"\n{name}  |  {image_size}x{image_size}{amp_tag}")
    print(f"  {'Batch':>5s} | {'FPS':>8s} | {'Latency (ms)':^22s} | {'Peak Mem (MB)':>13s}")
    print(f"  {'-'*5:>5s}-+-{'-'*8:>8s}-+-{'-'*22:^22s}-+-{'-'*13:>13s}")
    for r in results:
        print(f"  {r['batch_size']:5d} | {r['fps']:8.1f} | "
              f"{r['mean_ms']:7.2f} +/- {r['std_ms']:5.2f}      | {r['peak_mem_mb']:13.0f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS for DINO and HRNet-W32 models")
    parser.add_argument("--config_a", type=str, help="Config YAML for model A")
    parser.add_argument("--checkpoint_a", type=str, default=None, help="Checkpoint for model A (optional, random weights if omitted)")
    parser.add_argument("--name_a", type=str, default=None, help="Display name for model A")
    parser.add_argument("--config_b", type=str, help="Config YAML for model B")
    parser.add_argument("--checkpoint_b", type=str, default=None, help="Checkpoint for model B (optional)")
    parser.add_argument("--name_b", type=str, default=None, help="Display name for model B")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32], help="Batch sizes to benchmark")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Warmup iterations (discarded)")
    parser.add_argument("--bench_iters", type=int, default=200, help="Timed benchmark iterations")
    parser.add_argument("--amp", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile()")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", help="torch.compile mode")
    parser.add_argument("--only_a", action="store_true", help="Only benchmark model A")
    parser.add_argument("--only_b", action="store_true", help="Only benchmark model B")
    parser.add_argument("--trt_engines", type=str, nargs="+", default=[], help="TensorRT .engine files to benchmark")
    parser.add_argument("--trt_config", type=str, default=None, help="Config YAML for TRT engines (needed for image_size)")
    parser.add_argument("--trt_names", type=str, nargs="+", default=[], help="Display names for TRT engines (optional, defaults to filenames)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    has_trt = len(args.trt_engines) > 0
    if not args.config_a and not args.only_b and not has_trt:
        parser.error("--config_a is required unless --only_b or --trt_engines is set")
    if not args.config_b and not args.only_a and not has_trt:
        parser.error("--config_b is required unless --only_a or --trt_engines is set")
    if has_trt and not args.trt_config:
        parser.error("--trt_config is required when using --trt_engines")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA device found. Benchmarking on CPU (timings will be inaccurate).")

    all_results = {}
    models_to_bench = []

    if not args.only_b and args.config_a:
        with open(args.config_a) as f:
            cfg_a = yaml.safe_load(f)
        name_a = args.name_a or cfg_a["model"]["backbone"]
        models_to_bench.append(("a", name_a, cfg_a, args.checkpoint_a))

    if not args.only_a and args.config_b:
        with open(args.config_b) as f:
            cfg_b = yaml.safe_load(f)
        name_b = args.name_b or cfg_b["model"]["backbone"]
        models_to_bench.append(("b", name_b, cfg_b, args.checkpoint_b))

    print(f"Benchmark settings: warmup={args.warmup_iters}, iters={args.bench_iters}, "
          f"batch_sizes={args.batch_sizes}, amp={args.amp}, compile={args.compile}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for key, name, cfg, ckpt_path in models_to_bench:
        img_size = cfg["data"]["image_size"]
        input_shape = (3, img_size, img_size)

        print(f"\nLoading {name}...")
        t0 = time.time()
        model = build_model(cfg, ckpt_path, device)
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")

        params = count_parameters(model)

        if args.compile:
            print(f"  Compiling with mode={args.compile_mode}...")
            try:
                model = torch.compile(model, mode=args.compile_mode)
                # Extra warmup for compiled model
                x = torch.randn(1, *input_shape, device=device)
                for _ in range(10):
                    model(x)
                    torch.cuda.synchronize()
                del x
                torch.cuda.empty_cache()
                print("  Compilation done.")
            except Exception as e:
                print(f"  torch.compile failed ({e}), falling back to eager mode")

        results = benchmark_model(model, input_shape, args.batch_sizes, args.warmup_iters, args.bench_iters, args.amp)
        print_results(name, img_size, params, results, args.amp)

        all_results[key] = {
            "name": name,
            "image_size": img_size,
            "params": params,
            "amp": args.amp,
            "compiled": args.compile,
            "benchmarks": results,
        }

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # TensorRT engines
    if has_trt:
        with open(args.trt_config) as f:
            trt_cfg = yaml.safe_load(f)
        trt_img_size = trt_cfg["data"]["image_size"]

        for i, engine_path in enumerate(args.trt_engines):
            if i < len(args.trt_names):
                trt_name = args.trt_names[i]
            else:
                trt_name = f"TRT: {os.path.basename(engine_path)}"

            engine_size_mb = os.path.getsize(engine_path) / (1024 ** 2)
            print(f"\nLoading {trt_name} ({engine_size_mb:.1f} MB)...")
            t0 = time.time()
            # TRTModel is loaded inside benchmark_trt, just measure load time here
            results = benchmark_trt(engine_path, trt_img_size, args.batch_sizes,
                                    args.warmup_iters, args.bench_iters)
            load_time = time.time() - t0
            print_results(trt_name, trt_img_size, None, results)

            all_results[f"trt_{i}"] = {
                "name": trt_name,
                "engine_path": engine_path,
                "engine_size_mb": engine_size_mb,
                "image_size": trt_img_size,
                "benchmarks": results,
            }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
