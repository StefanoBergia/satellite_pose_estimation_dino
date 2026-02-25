#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=hrnet_w32_msssim
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# Create log directory
mkdir -p slurm/logs

# Activate environment
PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"

cd "${PROJECT_DIR}"

python train.py --config config_hrnet_w32.yaml --pretrained outputs_hrnet_w32/converted_model.pth
