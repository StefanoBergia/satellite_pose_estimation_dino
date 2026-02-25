#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=benchmark_fps
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# Create log directory
mkdir -p slurm/logs

# Activate environment
PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"

cd "${PROJECT_DIR}"

echo "=== FP32 Benchmark ==="
python benchmark_fps.py --config_a config.yaml --checkpoint_a outputs_keypoints_heatmap/best_model.pth --name_a "DINO-ViT-L/16" --config_b config_hrnet_w32.yaml --checkpoint_b outputs_hrnet_w32/converted_model.pth --name_b "HRNet-W32" --output benchmark_results_fp32.json

echo ""
echo "=== AMP FP16 Benchmark ==="
python benchmark_fps.py --config_a config.yaml --checkpoint_a outputs_keypoints_heatmap/best_model.pth --name_a "DINO-ViT-L/16" --config_b config_hrnet_w32.yaml --checkpoint_b outputs_hrnet_w32/converted_model.pth --name_b "HRNet-W32" --amp --output benchmark_results_fp16.json
