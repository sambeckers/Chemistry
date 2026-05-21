#!/bin/bash
#SBATCH --job-name=dump_accum
#SBATCH --output=logs_%A/dump_accum_%A_%a.out
#SBATCH --error=logs_%A/dump_accum_%A_%a.err
#SBATCH --array=0-31           # 32 tasks — adjust to match --n-tasks below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32     # matches MAX_WORKERS in the Python script
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# --- chemistry to process: Crich or Orich ---
CHEMISTRY=${CHEMISTRY:-Crich}
# Fraction of particles to keep (0.1, 0.5, 1.0). Override with --export=FRACTION=0.1
FRACTION=${FRACTION:-0.5}

N_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

echo "Array task ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}  chemistry=${CHEMISTRY}  fraction=${FRACTION}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

python /fred/oz304/beckers/Chemistry/pipeline/plotting/plot_stats.py \
    --mode       accumulate         \
    --chemistry  "${CHEMISTRY}"     \
    --task-id    "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks    "${N_TASKS}"        \
    --fraction   "${FRACTION}"