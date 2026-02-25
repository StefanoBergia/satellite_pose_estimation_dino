#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=HRNet_W32_SelfTrain
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# Create log directory
mkdir -p slurm/logs

# Activate environment
PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"

cd "${PROJECT_DIR}"

CHECKPOINT="${1:-outputs_hrnet_w32/converted_model.pth}"

# Common eval parameters (matching colleague's eval.job for reproducible SLAB scores)
EVAL_PARAMS="--gt_crop --crop_pnp --resize_first --kpt_extractor argmax \
    --reproj_error 8 --ransac_confidence 0.999 --ransac_iterations 100 \
    --t_ratio_max 20 --refine_lm 1 --refine_retrim 1 --no_conf_filter"

echo "=== HRNet-W32 Self-Training ==="
python -m domain_adaptation.option2_self_training.self_train \
    --config domain_adaptation/option2_self_training/config_self_train.yaml \
    --pretrained "${CHECKPOINT}" \
    --no_crop ${EVAL_PARAMS}
