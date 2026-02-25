#!/bin/bash
#SBATCH --job-name=export_dino_trt
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

echo "=========================================="
echo "Step 1: FP16 + force_all_fp32 (debug test)"
echo "=========================================="
python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth --build_trt --fp16 --force_all_fp32 --dynamic_batch 1 8 16 --output_trt dino_vitl16_fp16_allfp32.engine

echo ""
echo "Step 1b: Evaluate force_all_fp32 engine"
python evaluate_trt.py --engine dino_vitl16_fp16_allfp32.engine --config config.yaml --batch_size 8 --splits val

echo ""
echo "=========================================="
echo "Step 2: FP16 mixed precision (selective)"
echo "=========================================="
python export_dino_tensorrt.py --config config.yaml --checkpoint outputs_keypoints_heatmap/best_model.pth --build_trt --fp16 --dynamic_batch 1 8 16 --output_trt dino_vitl16_fp16.engine

echo ""
echo "Step 2b: Evaluate mixed precision engine"
python evaluate_trt.py --engine dino_vitl16_fp16.engine --config config.yaml --batch_size 8 --splits val

echo "Done"
