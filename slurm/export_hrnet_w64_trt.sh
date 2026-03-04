#!/bin/bash
#SBATCH --job-name=export_hrnet_w64_trt
#SBATCH --partition=A100
#SBATCH --nodelist=thor
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=50GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024

source .venv/bin/activate

W64_CKPT=/nfs/home/caracciolo/pose_hrnet/runs/2026-02-27_11-52-35_hrnet_augStrong_coord/checkpoints/hrnet_kpts_best.pth

echo "=========================================="
echo "Step 1: Convert colleague checkpoint to project format"
echo "=========================================="
python scripts/convert_colleague_ckpt.py --input $W64_CKPT --config config_hrnet_w64.yaml --output outputs_hrnet_w64/converted_model.pth

echo ""
echo "=========================================="
echo "Step 2: Export ONNX + build TRT FP16 engine"
echo "=========================================="
python export_hrnet_tensorrt.py --config config_hrnet_w64.yaml --checkpoint outputs_hrnet_w64/converted_model.pth --output_onnx outputs_hrnet_w64/hrnet_w64.onnx --output_trt outputs_hrnet_w64/hrnet_w64_fp16.engine --build_trt --fp16 --validate

echo "Done"
