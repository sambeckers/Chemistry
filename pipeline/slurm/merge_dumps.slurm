#!/bin/bash
# Merge all batch_*.h5 files into final dump_*.h5 files — one array task per dump.
#SBATCH --job-name=merge
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-0
#SBATCH --output=logs/logs_%A/merge_%A_%a.out
#SBATCH --error=logs/logs_%A/merge_%A_%a.err

set -euo pipefail

# Prefer submit-wrapper exported root; fallback is useful for manual debugging.
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

cd "${ROOT}"
echo "WORKFLOW_TASK|stage=merge_dumps|dump_index=${SLURM_ARRAY_TASK_ID}|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" "${ROOT}/scripts/merge_batches.py" \
	--config "${CONFIG_PATH}" \
	--dump-index "${SLURM_ARRAY_TASK_ID}"
status=$?
set -e
echo "WORKFLOW_TASK|stage=merge_dumps|dump_index=${SLURM_ARRAY_TASK_ID}|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
