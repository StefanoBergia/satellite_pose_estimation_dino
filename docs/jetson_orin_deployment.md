# Jetson Orin Deployment Guide

Export, evaluate, and benchmark HRNet-W32 and DINO ViT-L/16 TensorRT engines.

**Important**: TensorRT engines are not portable across GPU architectures. Engines must be built on the Jetson Orin itself (or on the same TRT version + GPU arch). ONNX files are portable.

## Prerequisites

Adjust dataset paths below to match the Jetson filesystem. On the server they are:
- Images: `/nfs/home/caracciolo/dataset/speedplus_yolo/images/val`
- Labels: `/nfs/home/caracciolo/dataset/speedplus_yolo/labels/val`

---

## 1. HRNet-W32

- Config: `config_hrnet_w32.yaml`
- Checkpoint: `outputs_hrnet_w32/converted_model.pth`
- Input: 512x512, no ImageNet normalization
- Heatmap: 128x128

### 1.1 Export ONNX

```
python export_hrnet_tensorrt.py --config config_hrnet_w32.yaml --checkpoint outputs_hrnet_w32/converted_model.pth --output_onnx hrnet_w32.onnx --dynamic_batch 1 8 16 --validate
```

### 1.2 Build TensorRT Engines

**FP16:**
```
python export_hrnet_tensorrt.py --config config_hrnet_w32.yaml --checkpoint outputs_hrnet_w32/converted_model.pth --build_trt --fp16 --dynamic_batch 1 8 16 --output_trt hrnet_w32_fp16.engine
```

**INT8 (with calibration):**
```
python export_hrnet_tensorrt.py --config config_hrnet_w32.yaml --checkpoint outputs_hrnet_w32/converted_model.pth --build_trt --int8 --dynamic_batch 1 8 16 --output_trt hrnet_w32_int8.engine --calib_images /nfs/home/caracciolo/dataset/speedplus_yolo/images/val --calib_labels /nfs/home/caracciolo/dataset/speedplus_yolo/labels/val --num_calib_images 200
```

### 1.3 Evaluate Accuracy (compare against PyTorch baseline)

Uses the same PnP parameters as `slurm/evaluate_hrnet_w32.sh`:

**FP16:**
```
python evaluate_trt.py --engine hrnet_w32_fp16.engine --config config_hrnet_w32.yaml --batch_size 16 --gt_crop --crop_pnp --resize_first --kpt_extractor argmax --reproj_error 12 --ransac_confidence 0.999 --ransac_iterations 500 --min_kpt_area 5000 --t_ratio_max 10 --refine_lm 1 --refine_retrim 1 --min_inliers_schedule "11,9,8,6,4" --rmse_inliers_thr 15 --no_conf_filter --output hrnet_w32_fp16_results.txt
```

**INT8:**
```
python evaluate_trt.py --engine hrnet_w32_int8.engine --config config_hrnet_w32.yaml --batch_size 16 --gt_crop --crop_pnp --resize_first --kpt_extractor argmax --reproj_error 12 --ransac_confidence 0.999 --ransac_iterations 500 --min_kpt_area 5000 --t_ratio_max 10 --refine_lm 1 --refine_retrim 1 --min_inliers_schedule "11,9,8,6,4" --rmse_inliers_thr 15 --no_conf_filter --output hrnet_w32_int8_results.txt
```

### 1.4 Benchmark FPS

```
python benchmark_fps.py --trt_engines hrnet_w32_int8.engine hrnet_w32_fp16.engine --trt_config config_hrnet_w32.yaml --trt_names "HRNet-W32 INT8" "HRNet-W32 FP16" --batch_sizes 1 4 8 16
```

---

## 2. DINO ViT-L/16

- Config: `config.yaml`
- Checkpoint: `outputs_keypoints_heatmap/best_model.pth`
- Input: 224x224, with ImageNet normalization
- Heatmap: 64x64 during training, **56x56 in TRT** (export script patches to native deconv output to avoid TRT Resize op issues)

> **FP16 precision note**: ViT attention logits and LayerNorm can overflow FP16 range (~65504), producing NaN outputs. The `--force_all_fp32` flag forces all float layers to FP32 while keeping the FP16 engine format. This is **required** for correct DINO FP16 inference. Without it, the engine produces all-NaN keypoints.
>
> **INT8 not supported**: INT8 calibration fails for the DINO ViT-L architecture (CUDA driver error during calibration). Use FP16 with `--force_all_fp32` instead.

### 2.1 Export ONNX + Build TensorRT Engine

The export script automatically patches the heatmap head from 64x64 to the native deconv output (56x56), removing the `F.interpolate` op that causes TRT shape inference failures.

**FP16 (recommended):**
```
python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth --build_trt --fp16 --force_all_fp32 --dynamic_batch 1 8 16 --output_trt dino_vitl16_fp16.engine
```

**ONNX only (for validation or manual TRT build):**
```
python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth --output_onnx dino_vitl16.onnx --dynamic_batch 1 8 16 --validate
```

### 2.2 Evaluate Accuracy (compare against PyTorch baseline)

Uses default PnP parameters matching `slurm/evaluate_all.sh` (no GT crop, softargmax, confidence pre-filtering ON):

```
python evaluate_trt.py --engine dino_vitl16_fp16.engine --config config.yaml --batch_size 8 --output dino_vitl16_fp16_results.txt
```

> **Batch size**: Must not exceed the engine max dynamic batch (16 with the commands above). Default is 16; use 8 on memory-constrained devices like Jetson.

### 2.3 Benchmark FPS

```
python benchmark_fps.py --trt_engines dino_vitl16_fp16.engine --trt_config config.yaml --trt_names "DINO ViT-L FP16" --batch_sizes 1 4 8 16
```

---

## 3. Combined Benchmark

HRNet and DINO have different image sizes (512 vs 224), so benchmark them separately:

```
python benchmark_fps.py --trt_engines hrnet_w32_int8.engine hrnet_w32_fp16.engine --trt_config config_hrnet_w32.yaml --trt_names "HRNet-W32 INT8" "HRNet-W32 FP16" --batch_sizes 1 4 8 16
```

```
python benchmark_fps.py --trt_engines dino_vitl16_fp16.engine --trt_config config.yaml --trt_names "DINO ViT-L FP16" --batch_sizes 1 4 8 16
```

---

## 4. Files to Copy to Jetson

Minimum files needed (no PyTorch/HuggingFace required for TRT inference):

**Shared:**
- `evaluate_trt.py`, `evaluate_robust.py`, `benchmark_fps.py`
- `src/dataset.py`, `src/transforms.py`, `src/utils.py`, `src/losses.py`
- `tango3Dpoints.json`, `camera.json`

**DINO:**
- `config.yaml`
- `model_dino.onnx` + `model_dino.onnx.data` (rebuild engine on Jetson from ONNX)

**HRNet-W32:**
- `config_hrnet_w32.yaml`
- `hrnet_w32.onnx` + `hrnet_w32.onnx.data` (rebuild engine on Jetson from ONNX)

> Rebuild engines on Jetson — do not copy `.engine` files from the server (different GPU arch).

---

## 5. Notes

- **Dynamic batch `1 8 16`**: min=1, optimal=8, max=16. Use `--batch_size 16` or less for evaluation.
- **Calibration cache**: Once generated (e.g. `hrnet_w32_int8_calib.cache`), re-export is faster. Copy the cache to Jetson if calibrating there.
- **Engine rebuild required**: If you update TensorRT version, engines must be rebuilt from ONNX.
- **Memory on Jetson**: If OOM on batch 16, reduce `--dynamic_batch 1 4 8` and use `--batch_size 4`.
- **"Missing scale and zero-point" warnings**: Normal during INT8 build. Those layers fall back to FP16/FP32 automatically.
- **DINO `--force_all_fp32`**: This flag enables FP16 weight storage but forces all compute to FP32. The speed benefit is marginal over a pure FP32 engine, but it avoids the ViT FP16 overflow. For real FP16 speedup on Jetson, use HRNet-W32 instead.
