#!/bin/bash
# Submit pipeline merge job with dated logs in scratch/work area.
#
# Usage:
#   ./workflow/submit_merge.sh
#   ./workflow/submit_merge.sh --config /path/to/pipeline_config.yaml
#
# Workflow summary:
# 1) Resolve config and runtime paths.
# 2) Query python binary, scratch root, and dump count from config.
# 3) Submit merge_dumps.slurm as a SLURM array job (one task per dump).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/merge_dumps.slurm"

CONFIG_PATH="${DEFAULT_CONFIG}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./workflow/submit_merge.sh [--config PATH]"
            exit 1
            ;;
    esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}"
    exit 1
fi

# Keep shell logic minimal by delegating config parsing to Python.
readarray -t INFO < <(
    /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()

sys.path.insert(0, str(pipeline_root / "scripts"))

from common import DumpSelection, PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))
dump_count = len(DumpSelection.target_dump_numbers(cfg))

print(python_bin)
print(scratch_root)
print(dump_count)
PY
)

if [[ "${#INFO[@]}" -lt 3 ]]; then
    echo "Error: failed to read merge submission metadata from config."
    exit 1
fi

PYTHON_BIN="${INFO[0]}"
SCRATCH_ROOT="${INFO[1]}"
DUMP_COUNT="${INFO[2]}"

if [[ -z "${DUMP_COUNT}" || "${DUMP_COUNT}" -lt 1 ]]; then
    echo "Error: invalid dump count '${DUMP_COUNT}'"
    exit 1
fi

MERGE_ARRAY_MAX=$((DUMP_COUNT - 1))
MERGE_ARRAY_SPEC="0-${MERGE_ARRAY_MAX}"

mkdir -p "${SCRATCH_ROOT}/logs"
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}"

echo "========================================="
echo "Chemistry HDF5 Merge Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Python          : ${PYTHON_BIN}"
echo "Dump count      : ${DUMP_COUNT} (array ${MERGE_ARRAY_SPEC})"
echo "Logs            : ${LOG_DIR}"

echo "Submitting..."
sbatch \
    --array="${MERGE_ARRAY_SPEC}" \
    --output="${LOG_DIR}/merge_%A_%a.out" \
    --error="${LOG_DIR}/merge_%A_%a.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN}" \
    "${SLURM_SCRIPT}"
