#!/bin/bash
# One array task = one pipeline batch index.
#SBATCH --job-name=batch
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
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

# ---------------------------------------------------------------------------
# Resolve real batch index
#
# Two modes:
#
#   Normal mode  (no PENDING_BATCH_FILE):
#     BATCH_INDEX = SLURM_ARRAY_TASK_ID + BATCH_INDEX_OFFSET
#     BATCH_INDEX_OFFSET is 0 for single-chunk runs, or the chunk's start
#     index for multi-chunk submissions.
#
#   Resume mode  (PENDING_BATCH_FILE is set):
#     BATCH_INDEX is read from line (SLURM_ARRAY_TASK_ID + BATCH_INDEX_OFFSET + 1)
#     of the pending-batch file written by submit_batch.sh --resume.
#     BATCH_INDEX_OFFSET is the line offset for this chunk (0, 2048, 4096, …).
# ---------------------------------------------------------------------------
BATCH_INDEX_OFFSET="${BATCH_INDEX_OFFSET:-0}"

if [[ -n "${PENDING_BATCH_FILE:-}" ]]; then
	# Resume mode: map task position → real batch index via the pending file.
	LINE_NUM=$(( SLURM_ARRAY_TASK_ID + BATCH_INDEX_OFFSET + 1 ))
	BATCH_INDEX=$(sed -n "${LINE_NUM}p" "${PENDING_BATCH_FILE}")
	if [[ -z "${BATCH_INDEX}" ]]; then
		echo "Error: no entry at line ${LINE_NUM} in PENDING_BATCH_FILE=${PENDING_BATCH_FILE}"
		echo "  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  BATCH_INDEX_OFFSET=${BATCH_INDEX_OFFSET}"
		exit 1
	fi
else
	# Normal mode: arithmetic offset.
	BATCH_INDEX=$(( SLURM_ARRAY_TASK_ID + BATCH_INDEX_OFFSET ))
fi

echo "BATCH_TASK|batch=${BATCH_INDEX}|phase=start|epoch=$(date +%s)"

# Unified workflow script executes one batch task in run mode.
set +e
"${ROOT}/workflow/batch.sh" --run-index "${BATCH_INDEX}"
status=$?
set -e

echo "BATCH_TASK|batch=${BATCH_INDEX}|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
