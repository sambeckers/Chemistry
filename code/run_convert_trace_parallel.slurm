#!/bin/bash
#SBATCH --job-name=evolving_model_parallel
#SBATCH --output=logs/chem_trace_parallel_%A_%a.out
#SBATCH --error=logs/chem_trace_parallel_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-14

set -euo pipefail

WORKDIR="/fred/oz304/beckers/Chemistry/code"
cd "$WORKDIR"

mkdir -p logs

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/MRP_env

echo "Starting job ${SLURM_JOB_ID:-NA}, array task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"

# One SLURM array task processes one selected particle ID.
python convert_trace__run_models_parallel.py \
  --pmf wind_v10 \
  --traces-subdir trace_output_with_av \
  --model-dir evolving_model \
  --max-particle-id 300000 \
  --start-index 2 \
  --n-select 15 \
  --use-slurm-array \
  --slurm-index-base 0
