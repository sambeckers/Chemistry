#!/bin/bash
# Generate one post-run summary report from completed batch array logs.
#SBATCH --job-name=summary
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/logs_%A/summary_%A.out
#SBATCH --error=logs/logs_%A/summary_%A.err

set -euo pipefail

if [[ -n "${PIPELINE_ROOT:-}" ]]; then
	ROOT="${PIPELINE_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
	echo "Error: PIPELINE_ROOT is not set and SLURM_SUBMIT_DIR is unavailable."
	exit 1
fi

PYTHON_BIN="${PIPELINE_PYTHON:-/fred/oz304/beckers/MRP_env/bin/python}"

if [[ -z "${BATCH_LOG_DIR:-}" ]]; then
	echo "Error: BATCH_LOG_DIR is not set."
	exit 1
fi

cd "${ROOT}"
exec "${PYTHON_BIN}" "${ROOT}/scripts/summarize_batch_logs.py"
