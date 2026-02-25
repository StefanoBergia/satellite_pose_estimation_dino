#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=HRNet_evaluate
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# Create log directory
mkdir -p slurm/logs

# Activate environment
PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"

cd "${PROJECT_DIR}"

CHECKPOINT="${1:-outputs_hrnet_heatmap/best_model.pth}"

echo "=== HRNet Robust Evaluation ==="
python evaluate_robust.py --checkpoint "${CHECKPOINT}"

echo ""
echo "=== HRNet TTA (TENT, 5 steps) ==="
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint "${CHECKPOINT}" \
    --method tent \
    --num_steps 5

echo ""
echo "=== HRNet TTA (NormAdapt) ==="
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint "${CHECKPOINT}" \
    --method norm_adapt
