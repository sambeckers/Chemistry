#!/bin/bash
# Gather phase: assemble one dump_XXXXX.h5 from all scatter_*.h5 files.
# One array task per dump.  Runs after the scatter array completes.
#
# Memory note: each task accumulates all particles for one dump across up to
# 10 000 scatter files.  With ~2 M particles per dump × ~500 datasets the
# in-memory footprint is roughly 8 GB; 32 GB gives headroom for spikes.
# Adjust --mem-per-cpu to match your per-dump particle count.
#
# The submit script overrides --array so the placeholder value below is never
# used in production.
#SBATCH --job-name=gather
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --tmp=20G
#SBATCH --array=0-0
#SBATCH --output=logs/logs_%A/gather_%A_%a.out
#SBATCH --error=logs/logs_%A/gather_%A_%a.err

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

REAL_INDEX=$(sed -n "$((SLURM_ARRAY_TASK_ID + INDEX_OFFSET + 1))p" "${PENDING_INDEX_FILE}")

if [[ -z "${REAL_INDEX}" ]]; then
    echo "Error: failed to resolve gather dump index"
    exit 1
fi

echo "WORKFLOW_TASK|stage=gather_dumps|dump_index=${SLURM_ARRAY_TASK_ID}|phase=start|epoch=$(date +%s)"
set +e
"${PYTHON_BIN}" "${ROOT}/scripts/gather_dumps.py" \
    --config "${CONFIG_PATH}" \
    --dump-index "${REAL_INDEX}" \
    --output-dir "${JOBFS}"
status=$?
set -e

if [[ "${status}" -eq 0 ]]; then
    shopt -s nullglob
    staged=("${JOBFS}"/dump_*.h5)
    shopt -u nullglob
    if [[ "${#staged[@]}" -ne 1 ]]; then
        echo "Error: expected exactly one dump_*.h5 in \$JOBFS, found ${#staged[@]}"
        status=1
    else
        cp "${staged[0]}" "${FINAL_OUTPUT_DIR}/"
    fi
fi

echo "WORKFLOW_TASK|stage=gather_dumps|dump_index=${SLURM_ARRAY_TASK_ID}|phase=end|epoch=$(date +%s)|status=${status}"
exit "${status}"