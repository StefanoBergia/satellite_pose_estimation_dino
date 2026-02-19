#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH  --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=Dino_v3_domain_generalization
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# Create log directory
mkdir -p slurm/logs

# Activate environment
PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"

cd "${PROJECT_DIR}"

python -m domain_adaptation.option3_domain_generalization.train_dg \
    --config domain_adaptation/option3_domain_generalization/config_dg.yaml \
    --pretrained outputs_keypoints_heatmap/best_model.pth
