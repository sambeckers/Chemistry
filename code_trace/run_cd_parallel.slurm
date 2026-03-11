#!/bin/bash
#SBATCH --job-name=cd_parallel
#SBATCH --output=logs/logs_%A/cd_parallel_%A_%a.out  # overridden by process_all_dumps.sh
#SBATCH --error=logs/logs_%A/cd_parallel_%A_%a.err   # overridden by process_all_dumps.sh
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --array=0-0  # overridden dynamically by process_all_dumps.sh

set -euo pipefail

WORKDIR="/fred/oz304/beckers/Chemistry/code_trace"
cd "$WORKDIR"

DATA_DIR="${DATA_DIR:-/fred/oz304/beckers/v20a25}"
WORK_DIR="${WORK_DIR:-/fred/oz304/beckers/v20a25_out}"
PREFIX="${PREFIX:-wind}"

# Read the dump number for this array task from the list written by process_all_dumps.sh
DUMP_LIST="${WORK_DIR}/dump_list.txt"
DUMP_NUM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$DUMP_LIST")

if [[ -z "$DUMP_NUM" ]]; then
    echo "Error: no dump number found for task ${SLURM_ARRAY_TASK_ID} in ${DUMP_LIST}"
    exit 1
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/magritte

echo "Starting job ${SLURM_ARRAY_JOB_ID:-NA}, array task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
echo "Processing dump ${DUMP_NUM}"

export DUMP_NUM="$DUMP_NUM"
export DATA_DIR="$DATA_DIR"
export WORK_DIR="$WORK_DIR"

python column_densities.py
