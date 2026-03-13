#!/bin/bash
# Build particle-ID cache in parallel over dumps and update pipeline_config.yaml.
#
# Usage:
#   ./submit_discover_particle_ids.sh
#   ./submit_discover_particle_ids.sh --config /path/to/pipeline_config.yaml
#   ./submit_discover_particle_ids.sh --max-concurrent 256

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
ARRAY_SLURM="${SCRIPT_DIR}/discover_particle_ids_array.slurm"
MERGE_SLURM="${SCRIPT_DIR}/discover_particle_ids_merge.slurm"

CONFIG_PATH="${DEFAULT_CONFIG}"
MAX_CONCURRENT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./submit_discover_particle_ids.sh [--config PATH] [--max-concurrent N]"
            exit 1
            ;;
    esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}"
    exit 1
fi

readarray -t INFO < <(
    /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()

sys.path.insert(0, str((pipeline_root / "scripts").resolve()))

from common import get_selected_dump_numbers, load_pipeline_config  # noqa: E402

cfg = load_pipeline_config(config_path)
python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = Path(cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))).resolve()
selected = get_selected_dump_numbers(cfg)

print(python_bin)
print(str(scratch_root))
print(len(selected))
for number in selected:
    print(number)
PY
)

if [[ "${#INFO[@]}" -lt 4 ]]; then
    echo "Error: failed to read discovery metadata from config."
    exit 1
fi

PYTHON_BIN="${INFO[0]}"
SCRATCH_ROOT="${INFO[1]}"
N_DUMPS="${INFO[2]}"

if [[ "${N_DUMPS}" -lt 1 ]]; then
    echo "Error: no dumps selected."
    exit 1
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="${SCRATCH_ROOT}/particle_discovery/run_${TIMESTAMP}"
DUMP_LIST_FILE="${RUN_DIR}/dump_numbers.txt"
DISCOVER_OUT_DIR="${RUN_DIR}/per_dump_ids"

mkdir -p "${RUN_DIR}" "${DISCOVER_OUT_DIR}" "${SCRATCH_ROOT}/logs"

printf '%s\n' "${INFO[@]:3}" > "${DUMP_LIST_FILE}"

ARRAY_MAX=$((N_DUMPS - 1))
ARRAY_SPEC="0-${ARRAY_MAX}"
if [[ -n "${MAX_CONCURRENT}" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}"

echo "========================================="
echo "Particle ID Discovery Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Dumps           : ${N_DUMPS} (array ${ARRAY_SPEC})"
echo "Run dir         : ${RUN_DIR}"
echo "Logs            : ${LOG_DIR}"

echo "Submitting discovery array..."
ARRAY_SUBMIT_OUTPUT=$(sbatch \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/discover_ids_%A_%a.out" \
    --error="${LOG_DIR}/discover_ids_%A_%a.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},DISCOVER_DUMP_LIST=${DUMP_LIST_FILE},DISCOVER_OUT_DIR=${DISCOVER_OUT_DIR}" \
    "${ARRAY_SLURM}")

ARRAY_JOB_ID=$(echo "${ARRAY_SUBMIT_OUTPUT}" | awk '{print $NF}')
if [[ -z "${ARRAY_JOB_ID}" ]]; then
    echo "Error: could not parse discovery array job id from: ${ARRAY_SUBMIT_OUTPUT}"
    exit 1
fi

echo "Array submitted: ${ARRAY_SUBMIT_OUTPUT}"

echo "Submitting merge job with dependency afterok:${ARRAY_JOB_ID}..."
MERGE_SUBMIT_OUTPUT=$(sbatch \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --output="${LOG_DIR}/discover_merge_%A.out" \
    --error="${LOG_DIR}/discover_merge_%A.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},DISCOVER_DUMP_LIST=${DUMP_LIST_FILE},DISCOVER_OUT_DIR=${DISCOVER_OUT_DIR}" \
    "${MERGE_SLURM}")

echo "Merge submitted: ${MERGE_SUBMIT_OUTPUT}"

echo "Done."
