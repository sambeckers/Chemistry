#!/bin/bash
# Submit batch array job(s) independently.
#
# This is the "batch only" counterpart to batch.sh.  It does NOT submit
# tracing, merge, or summary jobs.  Use --after to wire it to a previously
# submitted tracing job.
#
# Because tracing/merge/summary are submitted separately they do not consume
# slots from the per-user MaxSubmitPU quota.  Up to CHUNKS_PER_WAVE=5 chunks
# are queued per wave, but the total is clipped to MAX_SUBMIT_JOBS=10 000 so
# the last chunk in a wave is shrunk if needed (e.g. 4×2048 + 1×1808 = 10 000).
#
# Usage:
#   ./workflow/submit_batch.sh
#   ./workflow/submit_batch.sh --config /path/to/pipeline_config.yaml
#   ./workflow/submit_batch.sh --after <tracing_job_id>
#   ./workflow/submit_batch.sh --max-concurrent 256
#
# Usage (run mode — called by run_batch.slurm tasks):
#   ./workflow/submit_batch.sh --run-index 42
#
# Prints on exit:
#   Batch log dir: /path/to/logs_JOBID_DATETIME
#   Batch job IDs: 12346:12347:12348:12349:12350

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/run_batch.slurm"

MAX_ARRAY_SIZE=2048
CHUNKS_PER_WAVE=5    # attempt up to 5 chunks per wave
MAX_SUBMIT_JOBS=10000 # hard cap: clips last chunk in a wave if needed

CONFIG_PATH="${PIPELINE_CONFIG:-${DEFAULT_CONFIG}}"
MAX_CONCURRENT=""
RUN_INDEX=""
AFTER_JOB_ID=""

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
        --run-index)
            RUN_INDEX="$2"
            shift 2
            ;;
        --after)
            AFTER_JOB_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./workflow/submit_batch.sh [--config PATH] [--max-concurrent N] [--after JOB_ID] [--run-index INDEX]"
            exit 1
            ;;
    esac
done

if [[ -n "${RUN_INDEX}" ]]; then
    # Run mode: one SLURM task executes one batch index.
    PYTHON_BIN="${PIPELINE_PYTHON:-/fred/oz304/beckers/MRP_env/bin/python}"
    echo "Run mode: executing batch index ${RUN_INDEX}"
    exec "${PYTHON_BIN}" "${PIPELINE_ROOT}/scripts/run_batch_pipeline.py" \
        --config "${CONFIG_PATH}" \
        --batch-index "${RUN_INDEX}"
fi

# ----- Submit mode -----

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

from common import BatchPlanner, PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
if cfg.get("processing", {}).get("discovered_particle_count_all") is None:
    print("Error: cached particle discovery not found.", file=sys.stderr)
    print("Run: ./workflow/submit_discover_particle_ids.sh", file=sys.stderr)
    sys.exit(2)

count = BatchPlanner.batch_count(cfg)
python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))

print(count)
print(python_bin)
print(scratch_root)
PY
)

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

# Build chunk table (mirrors batch.sh logic exactly).
ARRAY_CHUNK_OFFSETS=()
ARRAY_CHUNK_SIZES=()
CHUNK_START=0
while [[ "${CHUNK_START}" -lt "${BATCH_COUNT}" ]]; do
    CHUNK_END=$(( CHUNK_START + MAX_ARRAY_SIZE - 1 ))
    if [[ "${CHUNK_END}" -ge "${BATCH_COUNT}" ]]; then
        CHUNK_END=$(( BATCH_COUNT - 1 ))
    fi
    ARRAY_CHUNK_OFFSETS+=("${CHUNK_START}")
    ARRAY_CHUNK_SIZES+=("$(( CHUNK_END - CHUNK_START + 1 ))")
    CHUNK_START=$(( CHUNK_END + 1 ))
done

echo "========================================="
echo "Chemistry HDF5 Batch Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Batch count     : ${BATCH_COUNT} (${#ARRAY_CHUNK_OFFSETS[@]} chunk(s) of up to ${MAX_ARRAY_SIZE}, offsets: ${ARRAY_CHUNK_OFFSETS[*]})"
echo "Chunks per wave : ${CHUNKS_PER_WAVE} (capped at ${MAX_SUBMIT_JOBS} total slots per wave)"
echo "Python          : ${PYTHON_BIN}"
if [[ -n "${AFTER_JOB_ID}" ]]; then
    echo "Tracing dep     : afterok:${AFTER_JOB_ID}"
fi

ARRAY_JOB_IDS=()
ARRAY_JOB_ID=""   # first chunk — used for LOG_DIR name
PREV_WAVE_LAST_JOB_ID=""
CHUNK_COUNT="${#ARRAY_CHUNK_OFFSETS[@]}"
i=0

while [[ "${i}" -lt "${CHUNK_COUNT}" ]]; do
    WAVE_JOB_IDS=()
    WAVE_END=$(( i + CHUNKS_PER_WAVE ))
    if [[ "${WAVE_END}" -gt "${CHUNK_COUNT}" ]]; then
        WAVE_END="${CHUNK_COUNT}"
    fi

    # Track slots submitted this wave so we never exceed MAX_SUBMIT_JOBS.
    WAVE_SLOTS_REMAINING="${MAX_SUBMIT_JOBS}"

    for (( j=i; j<WAVE_END; j++ )); do
        if [[ "${WAVE_SLOTS_REMAINING}" -le 0 ]]; then
            # Wave is full; remaining chunks carry over to the next wave.
            WAVE_END="${j}"
            break
        fi

        CHUNK_OFFSET="${ARRAY_CHUNK_OFFSETS[$j]}"
        CHUNK_SIZE="${ARRAY_CHUNK_SIZES[$j]}"

        # Clip this chunk so the wave total does not exceed MAX_SUBMIT_JOBS.
        if [[ "${CHUNK_SIZE}" -gt "${WAVE_SLOTS_REMAINING}" ]]; then
            CHUNK_SIZE="${WAVE_SLOTS_REMAINING}"
        fi

        CHUNK_ARRAY_MAX=$(( CHUNK_SIZE - 1 ))
        CHUNK_ARRAY_SPEC="0-${CHUNK_ARRAY_MAX}"
        if [[ -n "${MAX_CONCURRENT}" ]]; then
            CHUNK_ARRAY_SPEC="${CHUNK_ARRAY_SPEC}%${MAX_CONCURRENT}"
        fi

        if [[ -z "${PREV_WAVE_LAST_JOB_ID}" ]]; then
            # First wave: depend on the tracing job (if given), otherwise no dep.
            if [[ -n "${AFTER_JOB_ID}" ]]; then
                CHUNK_DEPENDENCY="afterok:${AFTER_JOB_ID}"
                DEPENDENCY_FLAG=(--dependency="${CHUNK_DEPENDENCY}")
            else
                DEPENDENCY_FLAG=()
            fi
        else
            CHUNK_DEPENDENCY="afterany:${PREV_WAVE_LAST_JOB_ID}"
            DEPENDENCY_FLAG=(--dependency="${CHUNK_DEPENDENCY}")
        fi

        CHUNK_SUBMIT_OUTPUT=$(sbatch \
            "${DEPENDENCY_FLAG[@]}" \
            --array="${CHUNK_ARRAY_SPEC}" \
            --output="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/batch_%A_%a.out" \
            --error="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/batch_%A_%a.err" \
            --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},BATCH_INDEX_OFFSET=${CHUNK_OFFSET}" \
            "${SLURM_SCRIPT}")
        echo "${CHUNK_SUBMIT_OUTPUT}"
        CHUNK_JOB_ID=$(echo "${CHUNK_SUBMIT_OUTPUT}" | awk '{print $NF}')
        if [[ -z "${CHUNK_JOB_ID}" ]]; then
            echo "Error: could not parse batch array job id from: ${CHUNK_SUBMIT_OUTPUT}"
            exit 1
        fi
        ARRAY_JOB_IDS+=("${CHUNK_JOB_ID}")
        WAVE_JOB_IDS+=("${CHUNK_JOB_ID}")
        if [[ -z "${ARRAY_JOB_ID}" ]]; then
            ARRAY_JOB_ID="${CHUNK_JOB_ID}"
        fi
        WAVE_SLOTS_REMAINING=$(( WAVE_SLOTS_REMAINING - CHUNK_SIZE ))
    done

    PREV_WAVE_LAST_JOB_ID="${WAVE_JOB_IDS[-1]}"
    i="${WAVE_END}"
done

LOG_DIR="${SCRATCH_ROOT}/logs/logs_${ARRAY_JOB_ID}_${DATETIME}"
mkdir -p "${LOG_DIR}"

# Write metadata so summary script can report on it.
cat > "${LOG_DIR}/workflow_submit_metadata.txt" <<META
discovery_needed=
tracing_job_id=${AFTER_JOB_ID}
batch_count=${BATCH_COUNT}
chunk_count=${#ARRAY_CHUNK_OFFSETS[@]}
META

ALL_BATCH_JOB_IDS=$(IFS=:; echo "${ARRAY_JOB_IDS[*]}")

echo "Batch log dir: ${LOG_DIR}"
echo "Batch job IDs: ${ALL_BATCH_JOB_IDS}"
