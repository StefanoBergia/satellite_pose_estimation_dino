#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=eval_holdout_cumulative
#SBATCH --mem=50GB
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

mkdir -p slurm/logs

PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

CHECKPOINT="outputs_self_training_incremental_cumulative/final_model.pth"

echo "=== Holdout-chunk evaluation — incremental cumulative ==="
python evaluate_robust.py --checkpoint "${CHECKPOINT}" \
    --splits lightbox sunlamp --test_split --splits_dir data/splits_holdout \
    --gt_crop --crop_pnp --resize_first --kpt_extractor argmax \
    --reproj_error 8 --ransac_confidence 0.999 --ransac_iterations 100 \
    --min_kpt_area 0 --t_ratio_max 20 \
    --refine_lm 1 --refine_retrim 1 --min_inliers_schedule "" \
    --rmse_inliers_thr 0 --no_conf_filter \
    --output "outputs_self_training_incremental_cumulative/results_holdout.txt"
