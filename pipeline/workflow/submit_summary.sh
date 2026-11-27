#!/bin/bash
# Submit the post-run summary job independently.
#
# Usage:
#   ./workflow/submit_summary.sh --log-dir /path/to/logs_JOBID_DATETIME
#   ./workflow/submit_summary.sh --log-dir PATH --array-job-id JOBID --after MERGE_JOB_ID
#
#   --log-dir PATH       Required. Batch log directory printed by submit_batch.sh.
#   --array-job-id ID    First batch array chunk job ID (display only in report).
#   --after JOB_ID       SLURM dependency: afterany:<id> (e.g. merge job ID).
#   --config PATH        Pipeline config (for python binary; default: pipeline_config.yaml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find a usable Python interpreter.
# Priority: PIPELINE_PYTHON env var > python3 on PATH > hard error.
# (In SLURM tasks, PIPELINE_PYTHON is always set via --export by the submit scripts.)
_bootstrap_python() {
    if [[ -n "${PIPELINE_PYTHON:-}" && -x "${PIPELINE_PYTHON}" ]]; then
        echo "${PIPELINE_PYTHON}"; return
    fi
    local by_path
    by_path=$(command -v python3 2>/dev/null || true)
    if [[ -n "${by_path}" && -x "${by_path}" ]]; then
        echo "${by_path}"; return
    fi
    echo "Error: Python not found. Export PIPELINE_PYTHON or activate MRP_env." >&2
    exit 1
}
BOOTSTRAP_PYTHON="$(_bootstrap_python)"

DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
SUMMARY_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/summarize_batches.slurm"

CONFIG_PATH="${PIPELINE_CONFIG:-${DEFAULT_CONFIG}}"
LOG_DIR=""
ARRAY_JOB_ID="unknown"
AFTER_JOB_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --array-job-id)
            ARRAY_JOB_ID="$2"
            shift 2
            ;;
        --after)
            AFTER_JOB_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./workflow/submit_summary.sh --log-dir PATH [--array-job-id ID] [--after JOB_ID] [--config PATH]"
            exit 1
            ;;
    esac
done

if [[ -z "${LOG_DIR}" ]]; then
    echo "Error: --log-dir is required."
    echo "Usage: ./workflow/submit_summary.sh --log-dir PATH [--array-job-id ID] [--after JOB_ID] [--config PATH]"
    exit 1
fi

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "Error: log directory not found: ${LOG_DIR}"
    exit 1
fi

if [[ ! -f "${SUMMARY_SLURM_SCRIPT}" ]]; then
    echo "Error: summary slurm script not found: ${SUMMARY_SLURM_SCRIPT}"
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}"
    exit 1
fi

PYTHON_BIN=$(
    ${BOOTSTRAP_PYTHON} - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(pipeline_root / "scripts"))

from common import PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
print(cfg.get("paths", {}).get("python", ""))
PY
)

echo "========================================="
echo "Chemistry HDF5 Summary Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Log dir         : ${LOG_DIR}"
echo "Array job ID    : ${ARRAY_JOB_ID}"
if [[ -n "${AFTER_JOB_ID}" ]]; then
    echo "Dependency      : afterany:${AFTER_JOB_ID}"
fi

DEPENDENCY_ARGS=()
if [[ -n "${AFTER_JOB_ID}" ]]; then
    DEPENDENCY_ARGS=(--dependency="afterany:${AFTER_JOB_ID}")
fi

sbatch \
    "${DEPENDENCY_ARGS[@]}" \
    --output="${LOG_DIR}/summary_%A.out" \
    --error="${LOG_DIR}/summary_%A.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},BATCH_LOG_DIR=${LOG_DIR},BATCH_ARRAY_JOB_ID=${ARRAY_JOB_ID}" \
    "${SUMMARY_SLURM_SCRIPT}"
