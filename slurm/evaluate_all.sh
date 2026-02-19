#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=thor
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=50GB
#SBATCH --job-name=Dino_v3_evaluate_all
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

mkdir -p slurm/logs

PROJECT_DIR=/nfs/home/bergia/Projects/Satellite_pose_estimation/satellite_pose_estimation_dino
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

EVAL="python evaluate_robust.py --splits val lightbox sunlamp --test_split --batch_size 64"

echo "========================================"
echo "  Baseline (heatmap)"
echo "========================================"
$EVAL --checkpoint outputs_keypoints_heatmap/best_model.pth \
      --output outputs_keypoints_heatmap/results_robust.txt

echo "========================================"
echo "  Option 3: Domain Generalization (DSU)"
echo "========================================"
$EVAL --checkpoint outputs_domain_generalization/best_model.pth \
      --output outputs_domain_generalization/results_robust.txt

echo "========================================"
echo "  Option 2: Self-Training (from baseline), iter3"
echo "========================================"
$EVAL --checkpoint outputs_self_training/iter3/best_model.pth \
      --output outputs_self_training/iter3/results_robust.txt

echo "========================================"
echo "  Option 2: Self-Training (from DG), iter1"
echo "========================================"
$EVAL --checkpoint outputs_self_training_dg/iter1/best_model.pth \
      --output outputs_self_training_dg/iter1/results_robust.txt

echo "========================================"
echo "  Option 4: TTA (tent, from baseline)"
echo "========================================"
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_keypoints_heatmap/best_model.pth \
    --method tent --num_steps 5

echo "========================================"
echo "  Option 4: TTA (tent, from DG)"
echo "========================================"
python -m domain_adaptation.option4_tta.evaluate_tta \
    --checkpoint outputs_domain_generalization/best_model.pth \
    --method tent --num_steps 5

echo "Done!"
