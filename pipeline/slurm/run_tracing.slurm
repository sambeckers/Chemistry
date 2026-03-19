#!/bin/bash
# Run one global tracing pass and pack per-batch tracing HDF5 files.
#SBATCH --job-name=tracing
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --output=logs/logs_%A/tracing_%A.out
#SBATCH --error=logs/logs_%A/tracing_%A.err

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

# Enable OpenMP multithreading for phantomanalysis to massively speed up tracing
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${ROOT}"
echo "WORKFLOW_TASK|stage=tracing|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" -u "${ROOT}/scripts/run_tracing.py" --config "${CONFIG_PATH}"
status=$?
set -e
echo "WORKFLOW_TASK|stage=tracing|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
