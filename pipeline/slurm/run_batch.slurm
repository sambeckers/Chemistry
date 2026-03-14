#!/bin/bash
# One array task = one pipeline batch index.
#SBATCH --job-name=chem_hdf5_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-0
#SBATCH --output=logs/logs_%A/chem_hdf5_batch_%A_%a.out
#SBATCH --error=logs/logs_%A/chem_hdf5_batch_%A_%a.err

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
# Unified workflow script executes one batch task in run mode.
"${ROOT}/workflow/batch.sh" --run-index "${SLURM_ARRAY_TASK_ID}"
