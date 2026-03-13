#!/bin/bash
# Submit pipeline merge job with dated logs in scratch/work area.
#
# Usage:
#   ./submit_merge.sh
#   ./submit_merge.sh --config /path/to/pipeline_config.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
SLURM_SCRIPT="${SCRIPT_DIR}/merge_dumps.slurm"

CONFIG_PATH="${DEFAULT_CONFIG}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./submit_merge.sh [--config PATH]"
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

sys.path.insert(0, str(pipeline_root / "scripts"))

from common import load_pipeline_config  # noqa: E402

cfg = load_pipeline_config(config_path)
python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))

print(python_bin)
print(scratch_root)
PY
)

if [[ "${#INFO[@]}" -lt 2 ]]; then
    echo "Error: failed to read merge submission metadata from config."
    exit 1
fi

PYTHON_BIN="${INFO[0]}"
SCRATCH_ROOT="${INFO[1]}"

mkdir -p "${SCRATCH_ROOT}/logs"
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}"

echo "========================================="
echo "Chemistry HDF5 Merge Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Python          : ${PYTHON_BIN}"
echo "Logs            : ${LOG_DIR}"

echo "Submitting..."
sbatch \
    --output="${LOG_DIR}/chem_hdf5_merge_%A.out" \
    --error="${LOG_DIR}/chem_hdf5_merge_%A.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN}" \
    "${SLURM_SCRIPT}"
