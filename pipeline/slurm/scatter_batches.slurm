#!/bin/bash
# Scatter phase: transpose batch_*.h5 files into dump-organised scatter_*.h5 files.
#
# Each array task processes SCATTER_TASKS_PER_JOB batch files serially, which
# keeps scheduler overhead low when individual batches are fast.  The default
# is 10 batches per task, giving ~1000 array tasks for a 10K-batch run.
#
# Memory note: batches are processed one at a time so peak RAM is still one
# batch (~3 GB decompressed).  16 GB gives comfortable headroom regardless of
# how many batches are packed into a task.
#
# Time note: scale --time with SCATTER_TASKS_PER_JOB.  At ~2 min/batch,
# 10 batches/task needs ~20 min; 4:00:00 is a safe default.
#SBATCH --job-name=scatter
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-0
#SBATCH --output=logs/logs_%A/scatter_%A_%a.out
#SBATCH --error=logs/logs_%A/scatter_%A_%a.err

set -euo pipefail

if [[ -n "${PIPELINE_ROOT:-}" ]]; then
    ROOT="${PIPELINE_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
    echo "Error: PIPELINE_ROOT is not set and SLURM_SUBMIT_DIR is unavailable."
    exit 1
fi

CONFIG_PATH="${PIPELINE_CONFIG:-${ROOT}/config/pipeline_config.yaml}"
PYTHON_BIN="${PIPELINE_PYTHON:-/fred/oz304/beckers/MRP_env/bin/python}"

# Number of batch files processed serially by this task.
# Set by the submitter via --export; falls back to 10.
TASKS_PER_JOB="${SCATTER_TASKS_PER_JOB:-10}"

# Global task index (accounts for wave chunking by the submitter).
GLOBAL_TASK_IDX=$(( SLURM_ARRAY_TASK_ID + INDEX_OFFSET ))

# Extract the corresponding slice of batch indices from the pending file.
# Pending file is 1-indexed for sed; each task owns TASKS_PER_JOB consecutive lines.
START_LINE=$(( GLOBAL_TASK_IDX * TASKS_PER_JOB + 1 ))
END_LINE=$(( START_LINE + TASKS_PER_JOB - 1 ))

mapfile -t REAL_INDICES < <(sed -n "${START_LINE},${END_LINE}p" "${PENDING_INDEX_FILE}")

if [[ "${#REAL_INDICES[@]}" -eq 0 ]]; then
    echo "Error: no batch indices found for global task ${GLOBAL_TASK_IDX} " \
         "(lines ${START_LINE}-${END_LINE} of ${PENDING_INDEX_FILE})"
    exit 1
fi

echo "Task ${GLOBAL_TASK_IDX}: processing ${#REAL_INDICES[@]} batch(es): ${REAL_INDICES[*]}"

cd "${ROOT}"
echo "WORKFLOW_TASK|stage=scatter_batches|global_task=${GLOBAL_TASK_IDX}|n_batches=${#REAL_INDICES[@]}|phase=start|epoch=$(date +%s)"

set +e
"${PYTHON_BIN}" "${ROOT}/scripts/scatter_batches.py" \
    --config "${CONFIG_PATH}" \
    --batch-indices "${REAL_INDICES[@]}"
status=$?
set -e

echo "WORKFLOW_TASK|stage=scatter_batches|global_task=${GLOBAL_TASK_IDX}|n_batches=${#REAL_INDICES[@]}|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"