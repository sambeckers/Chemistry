#!/bin/bash
# Serial particle-ID discovery: one job scans all selected dumps.
# Expected environment variables are exported by submit_discover_particle_ids.sh:
#   PIPELINE_ROOT, PIPELINE_CONFIG, PIPELINE_PYTHON, DISCOVER_OUT_DIR
#SBATCH --job-name=discover_ids_serial
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/logs_%A/discover_ids_serial_%A.out
#SBATCH --error=logs/logs_%A/discover_ids_serial_%A.err

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
DISCOVER_OUT_DIR="${DISCOVER_OUT_DIR:?DISCOVER_OUT_DIR is required}"

cd "${ROOT}"
echo "WORKFLOW_TASK|stage=particle_discovery_serial|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" "${ROOT}/scripts/discover_particle_ids_parallel.py" \
    --mode serial \
    --config "${CONFIG_PATH}" \
    --out-dir "${DISCOVER_OUT_DIR}" \
    --update-config
status=$?
set -e
echo "WORKFLOW_TASK|stage=particle_discovery_serial|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
