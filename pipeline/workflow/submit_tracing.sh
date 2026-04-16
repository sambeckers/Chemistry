#!/bin/bash
# Submit the tracing job independently.
#
# Usage:
#   ./workflow/submit_tracing.sh
#   ./workflow/submit_tracing.sh --config /path/to/pipeline_config.yaml
#   ./workflow/submit_tracing.sh --after <discover_merge_job_id>
#
# Prints "Tracing job ID: <id>" on the last line so the calling shell can
# capture it and pass it to submit_batch.sh --after.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
TRACING_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/run_tracing.slurm"

CONFIG_PATH="${PIPELINE_CONFIG:-${DEFAULT_CONFIG}}"
AFTER_JOB_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --after)
            AFTER_JOB_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./workflow/submit_tracing.sh [--config PATH] [--after JOB_ID]"
            exit 1
            ;;
    esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}"
    exit 1
fi

if [[ ! -f "${TRACING_SLURM_SCRIPT}" ]]; then
    echo "Error: tracing slurm script not found: ${TRACING_SLURM_SCRIPT}"
    exit 1
fi

# Delegate config parsing to Python.
readarray -t INFO < <(
    /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(pipeline_root / "scripts"))

from common import PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))

print(python_bin)
print(scratch_root)
PY
)

if [[ "${#INFO[@]}" -lt 2 ]]; then
    echo "Error: failed to read config metadata."
    exit 1
fi

PYTHON_BIN="${INFO[0]}"
SCRATCH_ROOT="${INFO[1]}"

mkdir -p "${SCRATCH_ROOT}/logs"
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
TRACE_LOG_DIR="${SCRATCH_ROOT}/logs/tracing_logs_${DATETIME}"
mkdir -p "${TRACE_LOG_DIR}"

echo "========================================="
echo "Chemistry HDF5 Tracing Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Python          : ${PYTHON_BIN}"
echo "Tracing logs    : ${TRACE_LOG_DIR}"
if [[ -n "${AFTER_JOB_ID}" ]]; then
    echo "Dependency      : afterok:${AFTER_JOB_ID}"
fi

DEPENDENCY_ARGS=()
if [[ -n "${AFTER_JOB_ID}" ]]; then
    DEPENDENCY_ARGS=(--dependency="afterok:${AFTER_JOB_ID}")
fi

TRACING_SUBMIT_OUTPUT=$(sbatch \
    "${DEPENDENCY_ARGS[@]}" \
    --output="${TRACE_LOG_DIR}/tracing_%A.out" \
    --error="${TRACE_LOG_DIR}/tracing_%A.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN}" \
    "${TRACING_SLURM_SCRIPT}")
echo "${TRACING_SUBMIT_OUTPUT}"

TRACING_JOB_ID=$(echo "${TRACING_SUBMIT_OUTPUT}" | awk '{print $NF}')
if [[ -z "${TRACING_JOB_ID}" ]]; then
    echo "Error: could not parse tracing job id from: ${TRACING_SUBMIT_OUTPUT}"
    exit 1
fi

echo "Tracing job ID: ${TRACING_JOB_ID}"
