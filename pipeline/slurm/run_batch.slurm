#!/bin/bash
#SBATCH --job-name=chem_hdf5_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-0
#SBATCH --output=logs/logs_%A/chem_hdf5_batch_%A_%a.out
#SBATCH --error=logs/logs_%A/chem_hdf5_batch_%A_%a.err

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

if [[ ! -f "${ROOT}/workflow/run_batch.sh" ]]; then
	echo "Error: workflow script not found: ${ROOT}/workflow/run_batch.sh"
	exit 1
fi

cd "${ROOT}"
"${ROOT}/workflow/run_batch.sh" "${SLURM_ARRAY_TASK_ID}"
