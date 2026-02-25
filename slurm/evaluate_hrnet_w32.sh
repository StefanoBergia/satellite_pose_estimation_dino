#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name=HRNet_W32_evaluate
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

echo "=== HRNet-W32 Robust Evaluation (GT crop + crop PnP + resize first + argmax) ==="
python evaluate_robust.py --checkpoint "${CHECKPOINT}" \
    --gt_crop --crop_pnp --resize_first --kpt_extractor argmax \
    --reproj_error 12 --ransac_confidence 0.999 --ransac_iterations 500 \
    --min_kpt_area 5000 --t_ratio_max 10 \
    --refine_lm 1 --refine_retrim 1 --min_inliers_schedule "11,9,8,6,4" \
    --rmse_inliers_thr 15 --no_conf_filter

echo ""
echo "=== HRNet-W32 Robust Evaluation (colleague-aligned: 8px reproj, 100 iters, gates off) ==="
python evaluate_robust.py --checkpoint "${CHECKPOINT}" \
    --gt_crop --crop_pnp --resize_first --kpt_extractor argmax \
    --reproj_error 8 --ransac_confidence 0.999 --ransac_iterations 100 \
    --min_kpt_area 0 --t_ratio_max 20 \
    --refine_lm 1 --refine_retrim 1 --min_inliers_schedule "" \
    --rmse_inliers_thr 0 --no_conf_filter \
    --output "$(dirname "${CHECKPOINT}")/results_robust_aligned.txt"

echo ""
echo "=== HRNet-W32 Robust Evaluation (no crop, baseline) ==="
python evaluate_robust.py --checkpoint "${CHECKPOINT}" --no_crop \
    --refine_lm 1 --refine_retrim 1 --min_inliers_schedule "11,9,8,6,4" \
    --output "$(dirname "${CHECKPOINT}")/results_robust_nocrop.txt"

echo ""
echo "=== HRNet-W32 TTA (TENT, 5 steps) ==="
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint "${CHECKPOINT}" \
    --method tent \
    --num_steps 5 \
    --no_crop

echo ""
echo "=== HRNet-W32 TTA (NormAdapt) ==="
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint "${CHECKPOINT}" \
    --method norm_adapt \
    --no_crop
