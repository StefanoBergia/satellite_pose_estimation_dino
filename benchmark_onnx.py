"""Benchmark ONNX model FPS using ONNX Runtime.

Usage:
    python benchmark_onnx.py --onnx model_dino.onnx --image_size 224
    python benchmark_onnx.py --onnx hrnet_w32.onnx --image_size 512
    python benchmark_onnx.py --onnx model_dino.onnx --image_size 224 --warmup 20 --iters 100
"""

import argparse
import statistics
import time

import numpy as np


def benchmark(onnx_path, image_size, batch_size, warmup_iters, bench_iters):
    import onnxruntime as ort

    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")

    # Prefer CUDA, fall back to CPU
    if "CUDAExecutionProvider" in providers:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"
        print("WARNING: CUDA not available, running on CPU (not representative)")

    print(f"Using provider: {provider}")
    sess = ort.InferenceSession(onnx_path, providers=[provider])

    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(batch_size, 3, image_size, image_size).astype(np.float32)

    print(f"\nModel:   {onnx_path}")
    print(f"Input:   ({batch_size}, 3, {image_size}, {image_size})")
    print(f"Warmup:  {warmup_iters} iters")
    print(f"Bench:   {bench_iters} iters")

    # Warmup
    print("\nWarming up...")
    for _ in range(warmup_iters):
        sess.run(None, {input_name: dummy})

    # Benchmark
    print("Benchmarking...")
    latencies = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies)
    median_ms = statistics.median(latencies)
    fps = batch_size / (mean_ms / 1000)

    print(f"\nResults (batch_size={batch_size}):")
    print(f"  Latency: {mean_ms:.2f} +/- {std_ms:.2f} ms  (median: {median_ms:.2f} ms)")
    print(f"  FPS:     {fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model FPS with ONNX Runtime")
    parser.add_argument("--onnx", type=str, required=True, help="Path to .onnx file")
    parser.add_argument("--image_size", type=int, required=True, help="Input image size (e.g. 224 or 512)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    benchmark(args.onnx, args.image_size, args.batch_size, args.warmup, args.iters)


if __name__ == "__main__":
    main()
