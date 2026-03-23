#!/bin/bash
# One array task = one pipeline batch index.
#SBATCH --job-name=batch
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-0
#SBATCH --output=logs/logs_%A/batch_%A_%a.out
#SBATCH --error=logs/logs_%A/batch_%A_%a.err

set -euo pipefail

# Keep root resolution compatible with both submit wrappers and direct sbatch use.
if [[ -n "${PIPELINE_ROOT:-}" ]]; then
	ROOT="${PIPELINE_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
	echo "Error: PIPELINE_ROOT is not set and SLURM_SUBMIT_DIR is unavailable."
	exit 1
fi

CONFIG_PATH="${PIPELINE_CONFIG:-${ROOT}/config/pipeline_config.yaml}"

if [[ ! -f "${ROOT}/workflow/batch.sh" ]]; then
	echo "Error: workflow script not found: ${ROOT}/workflow/batch.sh"
	exit 1
fi

cd "${ROOT}"

# BATCH_INDEX_OFFSET is exported by batch.sh when submitting chunked arrays.
# It is 0 for small runs that fit within MaxArraySize (no chunking needed).
BATCH_INDEX_OFFSET="${BATCH_INDEX_OFFSET:-0}"
BATCH_INDEX=$(( SLURM_ARRAY_TASK_ID + BATCH_INDEX_OFFSET ))

echo "BATCH_TASK|batch=${BATCH_INDEX}|phase=start|epoch=$(date +%s)"

# Unified workflow script executes one batch task in run mode.
set +e
"${ROOT}/workflow/batch.sh" --run-index "${BATCH_INDEX}"
status=$?
set -e

echo "BATCH_TASK|batch=${BATCH_INDEX}|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
