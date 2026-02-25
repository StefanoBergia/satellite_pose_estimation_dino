#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=ensemble_eval
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

mkdir -p slurm/logs

PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

HRNET_CKPT="${1:-outputs_hrnet_w32/converted_model.pth}"
DINO_CKPT="${2:-outputs_self_training_dg/final_model.pth}"

echo "=== Ensemble: HRNet + DINO keypoint fusion ==="
echo "  HRNet: ${HRNET_CKPT}"
echo "  DINO:  ${DINO_CKPT}"
echo ""

python evaluate_ensemble.py --hrnet_checkpoint "${HRNET_CKPT}" --dino_checkpoint "${DINO_CKPT}" --splits val lightbox sunlamp --test_split --batch_size 32 --refine_lm 1 --refine_retrim 1 --sanity_check 3 --output results_ensemble.txt
