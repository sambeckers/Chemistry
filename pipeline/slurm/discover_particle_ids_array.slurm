#!/bin/bash
# One array task = one PHANTOM dump scan.
# Expected environment variables are exported by submit_discover_particle_ids.sh:
#   PIPELINE_ROOT, PIPELINE_CONFIG, PIPELINE_PYTHON, DISCOVER_DUMP_LIST, DISCOVER_OUT_DIR
#SBATCH --job-name=discover_ids
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-0
#SBATCH --output=logs/logs_%A/discover_ids_%A_%a.out
#SBATCH --error=logs/logs_%A/discover_ids_%A_%a.err

set -euo pipefail

# Resolve pipeline root robustly for both direct and wrapper-driven submissions.
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

# SLURM array index maps 1-to-1 to line number in dump list file.
if [[ ! -f "${DUMP_LIST_FILE}" ]]; then
    echo "Error: dump list file not found: ${DUMP_LIST_FILE}"
    exit 1
fi

DUMP_NUMBER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${DUMP_LIST_FILE}")
if [[ -z "${DUMP_NUMBER}" ]]; then
    echo "Error: no dump number for task ${SLURM_ARRAY_TASK_ID} in ${DUMP_LIST_FILE}"
    exit 1
fi

cd "${ROOT}"
echo "WORKFLOW_TASK|stage=particle_discovery_worker|item=${SLURM_ARRAY_TASK_ID}|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" "${ROOT}/scripts/discover_particle_ids_parallel.py" \
    --mode worker \
    --config "${CONFIG_PATH}" \
    --dump-number "${DUMP_NUMBER}" \
    --out-dir "${DISCOVER_OUT_DIR}"
status=$?
set -e
echo "WORKFLOW_TASK|stage=particle_discovery_worker|item=${SLURM_ARRAY_TASK_ID}|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"
