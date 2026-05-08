#!/bin/bash
# Scatter phase: transpose each batch_*.h5 into a dump-organised scatter_*.h5.
# One array task per batch file.  Tasks are fully independent with no shared
# output — safe to run with high concurrency.
#
# Memory note: each task loads one complete batch file into RAM.  With ~400
# particles × 1600 dumps × ~500 datasets the decompressed footprint can reach
# ~3 GB per task; 16 GB gives comfortable headroom.  Adjust --mem-per-cpu if
# your batches are larger.
#SBATCH --job-name=scatter
#SBATCH --time=1:00:00
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

REAL_INDEX=$(sed -n "$((SLURM_ARRAY_TASK_ID + INDEX_OFFSET + 1))p" "${PENDING_INDEX_FILE}")

if [[ -z "${REAL_INDEX}" ]]; then
    echo "Error: failed to resolve scatter batch index"
    exit 1
fi

cd "${ROOT}"
echo "WORKFLOW_TASK|stage=scatter_batches|batch_index=${SLURM_ARRAY_TASK_ID}|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" "${ROOT}/scripts/scatter_batches.py" \
    --config "${CONFIG_PATH}" \
    --batch-index "${REAL_INDEX}"
status=$?
set -e
echo "WORKFLOW_TASK|stage=scatter_batches|batch_index=${SLURM_ARRAY_TASK_ID}|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
