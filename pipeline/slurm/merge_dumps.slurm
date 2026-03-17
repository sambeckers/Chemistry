#!/bin/bash
# Merge all batch_*.h5 files into final dump_*.h5 files.
#SBATCH --job-name=merge
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/logs_%A/merge_%A.out
#SBATCH --error=logs/logs_%A/merge_%A.err

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
echo "WORKFLOW_TASK|stage=merge_dumps|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" "${ROOT}/scripts/merge_batches.py" --config "${CONFIG_PATH}"
status=$?
set -e
echo "WORKFLOW_TASK|stage=merge_dumps|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
