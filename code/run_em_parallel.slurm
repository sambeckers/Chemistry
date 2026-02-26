#!/bin/bash
#SBATCH --job-name=em_parallel
#SBATCH --output=logs/logs_%A/em_parallel_%A_%a.out
#SBATCH --error=logs/logs_%A/em_parallel_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-14

set -euo pipefail

WORKDIR="/fred/oz304/beckers/Chemistry/code"
cd "$WORKDIR"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/MRP_env

echo "Starting job ${SLURM_JOB_ID:-NA}, array task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
echo "Chemistry type: ${CHEMISTRY_TYPE:-<default set in python script>}"

# --slurm enables non-interactive mode and single-particle selection via
# SLURM_ARRAY_TASK_ID.  Science settings (paths, particle selection, analysis
# flags, …) are configured inside run_chemistry_models.py — keep --array
# in sync with n_select defined there.
python run_evolving_models.py --parallel
