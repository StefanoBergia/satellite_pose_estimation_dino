#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH  --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=self_train_incremental
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# Create log directory
mkdir -p slurm/logs

# Activate environment
PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"

cd "${PROJECT_DIR}"

python -m domain_adaptation.option2_self_training.self_train \
    --config domain_adaptation/option2_self_training/config_self_train_incremental.yaml \
    --pretrained outputs_domain_generalization_mssim/checkpoint_epoch030.pth --output_dir ./outputs_self_training_incremental_sequential
