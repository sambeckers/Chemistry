#!/bin/bash
# Build particle-ID cache in parallel over dumps and update pipeline_config.yaml.
#
# Usage:
#   ./workflow/submit_discover_particle_ids.sh
#   ./workflow/submit_discover_particle_ids.sh --config /path/to/pipeline_config.yaml
#   ./workflow/submit_discover_particle_ids.sh --max-concurrent 256
#   ./workflow/submit_discover_particle_ids.sh --serial
#
# Workflow summary:
# 1) Query config for selected dumps, python binary, and scratch root.
# 2) Materialize run directory with dump list and per-dump output folder.
# 3) Submit worker array job.
# 4) Submit merge job with dependency afterok:<array_job_id>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
ARRAY_SLURM="${PIPELINE_ROOT}/slurm/discover_particle_ids_array.slurm"
MERGE_SLURM="${PIPELINE_ROOT}/slurm/discover_particle_ids_merge.slurm"
SERIAL_SLURM="${PIPELINE_ROOT}/slurm/discover_particle_ids_serial.slurm"

CONFIG_PATH="${DEFAULT_CONFIG}"
MAX_CONCURRENT=""
SERIAL_MODE=0
SERIAL_MODE_SET=0

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
        --serial)
            SERIAL_MODE=1
            SERIAL_MODE_SET=1
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./workflow/submit_discover_particle_ids.sh [--config PATH] [--max-concurrent N] [--serial]"
            exit 1
            ;;
    esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}"
    exit 1
fi

# Python helper prints values as lines to keep shell parsing robust.
readarray -t INFO < <(
    /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()

sys.path.insert(0, str((pipeline_root / "scripts").resolve()))

from common import DumpSelection, PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = Path(cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))).resolve()
selected = DumpSelection.selected_dump_numbers(cfg)
discovery_mode = str(cfg.get("processing", {}).get("discovery_mode", "array")).strip().lower()
if discovery_mode not in {"array", "serial"}:
    raise ValueError("processing.discovery_mode must be 'array' or 'serial'")

print(python_bin)
print(str(scratch_root))
print(discovery_mode)
print(len(selected))
for number in selected:
    print(number)
PY
)

if [[ "${#INFO[@]}" -lt 5 ]]; then
    echo "Error: failed to read discovery metadata from config."
    exit 1
fi

PYTHON_BIN="${INFO[0]}"
SCRATCH_ROOT="${INFO[1]}"
DISCOVERY_MODE="${INFO[2]}"
N_DUMPS="${INFO[3]}"

if [[ "${SERIAL_MODE_SET}" -eq 0 && "${DISCOVERY_MODE}" == "serial" ]]; then
    SERIAL_MODE=1
fi

if [[ "${N_DUMPS}" -lt 1 ]]; then
    echo "Error: no dumps selected."
    exit 1
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="${SCRATCH_ROOT}/particle_discovery/run_${TIMESTAMP}"
DUMP_LIST_FILE="${RUN_DIR}/dump_numbers.txt"
DISCOVER_OUT_DIR="${RUN_DIR}/per_dump_ids"

mkdir -p "${RUN_DIR}" "${DISCOVER_OUT_DIR}" "${SCRATCH_ROOT}/logs"

printf '%s\n' "${INFO[@]:4}" > "${DUMP_LIST_FILE}"

ARRAY_MAX=$((N_DUMPS - 1))
ARRAY_SPEC="0-${ARRAY_MAX}"
if [[ -n "${MAX_CONCURRENT}" ]]; then
    # Optional concurrency cap for large dump lists.
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}"

echo "========================================="
echo "Particle ID Discovery Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
if [[ "${SERIAL_MODE}" -eq 1 ]]; then
    echo "Dumps           : ${N_DUMPS} (serial)"
else
    echo "Dumps           : ${N_DUMPS} (array ${ARRAY_SPEC})"
fi
echo "Run dir         : ${RUN_DIR}"
echo "Logs            : ${LOG_DIR}"

if [[ "${SERIAL_MODE}" -eq 1 ]]; then
    echo "Submitting serial discovery job..."
    SERIAL_SUBMIT_OUTPUT=$(sbatch \
        --output="${LOG_DIR}/discover_ids_serial_%A.out" \
        --error="${LOG_DIR}/discover_ids_serial_%A.err" \
        --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},DISCOVER_OUT_DIR=${DISCOVER_OUT_DIR}" \
        "${SERIAL_SLURM}")

    echo "Serial discovery submitted: ${SERIAL_SUBMIT_OUTPUT}"
    echo "Done."
    exit 0
fi

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
