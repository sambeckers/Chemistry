#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PIPELINE_CONFIG:-${PIPELINE_ROOT}/config/pipeline_config.yaml}"
PYTHON_BIN="${PIPELINE_PYTHON:-/fred/oz304/beckers/MRP_env/bin/python}"
BATCH_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ -z "${BATCH_INDEX}" ]]; then
    echo "Error: batch index is required (argument or SLURM_ARRAY_TASK_ID)."
    exit 1
fi

exec "${PYTHON_BIN}" "${PIPELINE_ROOT}/scripts/run_batch_pipeline.py" \
    --config "${CONFIG_PATH}" \
    --batch-index "${BATCH_INDEX}"
