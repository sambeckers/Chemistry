#!/bin/bash
# Submit pipeline batch jobs with dynamic array size and dated logs.
#
# Usage:
#   ./submit_batch.sh
#   ./submit_batch.sh --config /path/to/pipeline_config.yaml
#   ./submit_batch.sh --max-concurrent 256

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
SLURM_SCRIPT="${SCRIPT_DIR}/run_batch.slurm"

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
            echo "Usage: ./submit_batch.sh [--config PATH] [--max-concurrent N]"
            exit 1
            ;;
    esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}"
    exit 1
fi

DISCOVERY_NEEDED=0
if ! /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str((pipeline_root / "scripts").resolve()))

from common import load_pipeline_config, load_cached_particle_ids  # noqa: E402

cfg = load_pipeline_config(config_path)
cache_ids = load_cached_particle_ids(cfg)
count_all = cfg.get("processing", {}).get("discovered_particle_count_all")
if cache_ids is None or count_all is None:
    sys.exit(3)
if int(len(cache_ids)) != int(count_all):
    sys.exit(4)
PY
then
    DISCOVERY_NEEDED=1
fi

if [[ "${DISCOVERY_NEEDED}" -eq 1 ]]; then
    echo "Cached particle discovery is missing/stale for current config."
    echo "Rebuilding cache now via submit_discover_particle_ids.sh ..."
    "${SCRIPT_DIR}/submit_discover_particle_ids.sh" --config "${CONFIG_PATH}"
    echo "Discovery jobs submitted. Re-run submit_batch.sh after discovery merge completes."
    exit 0
fi

set +e
INFO_RAW=$(
    /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()

sys.path.insert(0, str(pipeline_root / "scripts"))

from common import get_batch_count, load_pipeline_config  # noqa: E402

cfg = load_pipeline_config(config_path)
if cfg.get("processing", {}).get("discovered_particle_count_all") is None:
    print("Error: cached particle discovery not found.", file=sys.stderr)
    print("Run: ./submit_discover_particle_ids.sh", file=sys.stderr)
    sys.exit(2)

count = get_batch_count(cfg)

python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))

print(count)
print(python_bin)
print(scratch_root)
PY
)
STATUS=$?
set -e

if [[ ${STATUS} -ne 0 ]]; then
    exit ${STATUS}
fi

readarray -t INFO <<< "${INFO_RAW}"

if [[ "${#INFO[@]}" -lt 3 ]]; then
    echo "Error: failed to read batch submission metadata from config."
    exit 1
fi

BATCH_COUNT="${INFO[0]}"
PYTHON_BIN="${INFO[1]}"
SCRATCH_ROOT="${INFO[2]}"

if [[ -z "${BATCH_COUNT}" || "${BATCH_COUNT}" -lt 1 ]]; then
    echo "Error: invalid batch count '${BATCH_COUNT}'"
    exit 1
fi

mkdir -p "${SCRATCH_ROOT}/logs"
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}"

ARRAY_MAX=$((BATCH_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_MAX}"
if [[ -n "${MAX_CONCURRENT}" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

echo "========================================="
echo "Chemistry HDF5 Batch Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Batch count     : ${BATCH_COUNT} (array ${ARRAY_SPEC})"
echo "Python          : ${PYTHON_BIN}"
echo "Logs            : ${LOG_DIR}"

echo "Submitting..."
sbatch \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/chem_hdf5_batch_%A_%a.out" \
    --error="${LOG_DIR}/chem_hdf5_batch_%A_%a.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN}" \
    "${SLURM_SCRIPT}"
