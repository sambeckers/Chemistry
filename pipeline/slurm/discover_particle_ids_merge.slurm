#!/bin/bash
# Merge step for particle-ID discovery.
# This job depends on successful completion of discovery array tasks.
#SBATCH --job-name=discover_merge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/logs_%A/discover_merge_%A.out
#SBATCH --error=logs/logs_%A/discover_merge_%A.err

set -euo pipefail

# Use exported PIPELINE_ROOT when available to avoid ambiguity in copied job scripts.
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
DUMP_LIST_FILE="${DISCOVER_DUMP_LIST:?DISCOVER_DUMP_LIST is required}"
DISCOVER_OUT_DIR="${DISCOVER_OUT_DIR:?DISCOVER_OUT_DIR is required}"

cd "${ROOT}"
echo "WORKFLOW_TASK|stage=particle_discovery_merge|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" "${ROOT}/scripts/discover_particle_ids_parallel.py" \
    --mode merge \
    --config "${CONFIG_PATH}" \
    --dump-list "${DUMP_LIST_FILE}" \
    --out-dir "${DISCOVER_OUT_DIR}" \
    --update-config
status=$?
set -e
echo "WORKFLOW_TASK|stage=particle_discovery_merge|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
