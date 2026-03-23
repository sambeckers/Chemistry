#!/bin/bash
# Unified batch workflow script.
#
# This script combines two previous responsibilities:
# 1) Submit mode: compute array size and submit run_batch.slurm.
# 2) Run mode: execute one batch index (used by run_batch.slurm tasks).
#
# Usage (submit mode):
#   ./workflow/batch.sh
#   ./workflow/batch.sh --config /path/to/pipeline_config.yaml
#   ./workflow/batch.sh --max-concurrent 256
#
# Usage (run mode, usually from SLURM):
#   ./workflow/batch.sh --run-index 42

set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${WORKFLOW_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"
SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/run_batch.slurm"
MERGE_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/merge_dumps.slurm"
SUMMARY_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/summarize_batches.slurm"
TRACING_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/run_tracing.slurm"
DISCOVER_ARRAY_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/discover_particle_ids_array.slurm"
DISCOVER_MERGE_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/discover_particle_ids_merge.slurm"
DISCOVER_SERIAL_SLURM_SCRIPT="${PIPELINE_ROOT}/slurm/discover_particle_ids_serial.slurm"

TOTAL_STEPS=9
MAX_ARRAY_SIZE=2048
CURRENT_STEP=0
STEP_START_EPOCH=0

_render_progress_bar() {
    local current="$1"
    local total="$2"
    local width=30
    local filled=0
    local percent=0
    if [[ "${total}" -gt 0 ]]; then
        filled=$((current * width / total))
        percent=$((current * 100 / total))
    fi
    local empty=$((width - filled))
    printf "["
    local i
    for ((i = 0; i < filled; i++)); do
        printf "#"
    done
    for ((i = 0; i < empty; i++)); do
        printf "-"
    done
    printf "] %3d%%" "${percent}"
}

step_start() {
    local label="$1"
    CURRENT_STEP=$((CURRENT_STEP + 1))
    STEP_START_EPOCH=$(date +%s)
    printf "%s  Step %d/%d  %s\n" "$(_render_progress_bar "${CURRENT_STEP}" "${TOTAL_STEPS}")" "${CURRENT_STEP}" "${TOTAL_STEPS}" "${label}"
}

step_done() {
    local finished_epoch
    finished_epoch=$(date +%s)
    local elapsed=$((finished_epoch - STEP_START_EPOCH))
    printf "             Completed in %ss\n" "${elapsed}"
}

step_note() {
    local message="$1"
    printf "             %s\n" "${message}"
}

CONFIG_PATH="${PIPELINE_CONFIG:-${DEFAULT_CONFIG}}"
MAX_CONCURRENT=""
RUN_INDEX=""

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
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./workflow/batch.sh [--config PATH] [--max-concurrent N] [--run-index INDEX]"
            exit 1
            ;;
    esac
done

if [[ -n "${RUN_INDEX}" ]]; then
    # Run mode: one task executes one batch index.
    PYTHON_BIN="${PIPELINE_PYTHON:-/fred/oz304/beckers/MRP_env/bin/python}"
    echo "Run mode: executing batch index ${RUN_INDEX}"
    exec "${PYTHON_BIN}" "${PIPELINE_ROOT}/scripts/run_batch_pipeline.py" \
        --config "${CONFIG_PATH}" \
        --batch-index "${RUN_INDEX}"
fi

# Submit mode starts here.
echo "Starting batch submission workflow..."
step_start "Validating configuration"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}"
    exit 1
fi
step_note "Config: ${CONFIG_PATH}"
step_done

step_start "Checking particle-ID discovery cache"
DISCOVERY_NEEDED=0
if ! /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str((pipeline_root / "scripts").resolve()))

from common import ParticleIdStore, PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
cache_ids = ParticleIdStore.load_cached(cfg)
count_all = cfg.get("processing", {}).get("discovered_particle_count_all")
if cache_ids is None or count_all is None:
    sys.exit(3)
if int(len(cache_ids)) != int(count_all):
    sys.exit(4)
PY
then
    DISCOVERY_NEEDED=1
    TOTAL_STEPS=10
fi
step_done

step_start "Loading batch submission metadata"
set +e
INFO_RAW=$(
    /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()

sys.path.insert(0, str(pipeline_root / "scripts"))

from common import BatchPlanner, DumpSelection, PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
if cfg.get("processing", {}).get("discovered_particle_count_all") is None:
    print("Error: cached particle discovery not found.", file=sys.stderr)
    print("Run: ./workflow/submit_discover_particle_ids.sh", file=sys.stderr)
    sys.exit(2)

count = BatchPlanner.batch_count(cfg)
discovery_mode = str(cfg.get("processing", {}).get("discovery_mode", "array")).strip().lower()
if discovery_mode not in {"array", "serial"}:
    raise ValueError("processing.discovery_mode must be 'array' or 'serial'")

python_bin = cfg.get("paths", {}).get("python", "/fred/oz304/beckers/MRP_env/bin/python")
scratch_root = cfg.get("paths", {}).get("scratch_root", str(pipeline_root / "work"))

dump_count = len(DumpSelection.target_dump_numbers(cfg))

print(count)
print(python_bin)
print(scratch_root)
print(discovery_mode)
print(dump_count)
PY
)
STATUS=$?
set -e

if [[ ${STATUS} -ne 0 ]]; then
    exit ${STATUS}
fi
step_done

readarray -t INFO <<< "${INFO_RAW}"
if [[ "${#INFO[@]}" -lt 5 ]]; then
    echo "Error: failed to read batch submission metadata from config."
    exit 1
fi

BATCH_COUNT="${INFO[0]}"
PYTHON_BIN="${INFO[1]}"
SCRATCH_ROOT="${INFO[2]}"
DISCOVERY_MODE="${INFO[3]}"
DUMP_COUNT="${INFO[4]}"

if [[ -z "${BATCH_COUNT}" || "${BATCH_COUNT}" -lt 1 ]]; then
    echo "Error: invalid batch count '${BATCH_COUNT}'"
    exit 1
fi

if [[ -z "${DUMP_COUNT}" || "${DUMP_COUNT}" -lt 1 ]]; then
    echo "Error: invalid dump count '${DUMP_COUNT}'"
    exit 1
fi

step_start "Preparing SLURM array and logging paths"
mkdir -p "${SCRATCH_ROOT}/logs"
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR=""
TRACE_LOG_DIR="${SCRATCH_ROOT}/logs/tracing_logs_${DATETIME}"
mkdir -p "${TRACE_LOG_DIR}"

# Split into chunks of MAX_ARRAY_SIZE. Each chunk submits array 0-N with an
# offset exported as BATCH_INDEX_OFFSET so run_batch.slurm computes the real index.
ARRAY_CHUNK_OFFSETS=()   # start index of each chunk
ARRAY_CHUNK_SIZES=()     # number of tasks in each chunk
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
ARRAY_SPEC="0-$(( MAX_ARRAY_SIZE - 1 )) x${#ARRAY_CHUNK_OFFSETS[@]} chunks"  # display only

MERGE_ARRAY_MAX=$((DUMP_COUNT - 1))
MERGE_ARRAY_SPEC="0-${MERGE_ARRAY_MAX}"
if [[ -n "${MAX_CONCURRENT}" ]]; then
    MERGE_ARRAY_SPEC="${MERGE_ARRAY_SPEC}%${MAX_CONCURRENT}"
fi
step_done

step_start "Printing submission summary"
echo "========================================="
echo "Chemistry HDF5 End-to-End Submit"
echo "========================================="
echo "Pipeline root   : ${PIPELINE_ROOT}"
echo "Config          : ${CONFIG_PATH}"
echo "Batch count     : ${BATCH_COUNT} (${#ARRAY_CHUNK_OFFSETS[@]} chunk(s) of up to ${MAX_ARRAY_SIZE}, offsets: ${ARRAY_CHUNK_OFFSETS[*]})"
echo "Dump count      : ${DUMP_COUNT} (merge array ${MERGE_ARRAY_SPEC})"
echo "Python          : ${PYTHON_BIN}"
echo "Logs root       : ${SCRATCH_ROOT}/logs/"
if [[ "${DISCOVERY_NEEDED}" -eq 1 ]]; then
    echo "Particle IDs    : discovery required (will submit discover+merge jobs)"
    echo "Discovery mode  : ${DISCOVERY_MODE}"
else
    echo "Particle IDs    : cache valid (discovery step skipped)"
fi
echo "Batch logs      : ${SCRATCH_ROOT}/logs/logs_<array_job_id>_${DATETIME}"
echo "Tracing logs    : ${TRACE_LOG_DIR}"
step_done

DISCOVER_ARRAY_JOB_ID=""
DISCOVER_MERGE_JOB_ID=""

if [[ "${DISCOVERY_NEEDED}" -eq 1 ]]; then
    step_start "Submitting particle discovery jobs"
    readarray -t DISCOVERY_INFO < <(
        /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str((pipeline_root / "scripts").resolve()))

from common import DumpSelection, PipelineConfigManager  # noqa: E402

cfg = PipelineConfigManager.load(config_path)
selected = DumpSelection.selected_dump_numbers(cfg)
for number in selected:
    print(number)
PY
    )

    if [[ "${#DISCOVERY_INFO[@]}" -lt 1 ]]; then
        echo "Error: no dump numbers resolved for particle discovery."
        exit 1
    fi

    DISCOVERY_RUN_DIR="${SCRATCH_ROOT}/particle_discovery/run_${DATETIME}"
    DISCOVER_DUMP_LIST="${DISCOVERY_RUN_DIR}/dump_numbers.txt"
    DISCOVER_OUT_DIR="${DISCOVERY_RUN_DIR}/per_dump_ids"
    mkdir -p "${DISCOVERY_RUN_DIR}" "${DISCOVER_OUT_DIR}"
    if [[ "${DISCOVERY_MODE}" == "array" ]]; then
        printf '%s\n' "${DISCOVERY_INFO[@]}" > "${DISCOVER_DUMP_LIST}"
    fi

    if [[ "${DISCOVERY_MODE}" == "serial" ]]; then
        DISCOVER_SERIAL_SUBMIT_OUTPUT=$(sbatch \
            --output="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/discover_ids_serial_%A.out" \
            --error="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/discover_ids_serial_%A.err" \
            --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},DISCOVER_OUT_DIR=${DISCOVER_OUT_DIR}" \
            "${DISCOVER_SERIAL_SLURM_SCRIPT}")
        echo "${DISCOVER_SERIAL_SUBMIT_OUTPUT}"
        DISCOVER_MERGE_JOB_ID=$(echo "${DISCOVER_SERIAL_SUBMIT_OUTPUT}" | awk '{print $NF}')
        if [[ -z "${DISCOVER_MERGE_JOB_ID}" ]]; then
            echo "Error: could not parse discovery serial job id from: ${DISCOVER_SERIAL_SUBMIT_OUTPUT}"
            exit 1
        fi
    else
        DISCOVER_ARRAY_MAX=$(( ${#DISCOVERY_INFO[@]} - 1 ))
        DISCOVER_ARRAY_SPEC="0-${DISCOVER_ARRAY_MAX}"
        if [[ -n "${MAX_CONCURRENT}" ]]; then
            DISCOVER_ARRAY_SPEC="${DISCOVER_ARRAY_SPEC}%${MAX_CONCURRENT}"
        fi

        DISCOVER_ARRAY_SUBMIT_OUTPUT=$(sbatch \
            --array="${DISCOVER_ARRAY_SPEC}" \
            --output="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/discover_ids_%A_%a.out" \
            --error="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/discover_ids_%A_%a.err" \
            --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},DISCOVER_DUMP_LIST=${DISCOVER_DUMP_LIST},DISCOVER_OUT_DIR=${DISCOVER_OUT_DIR}" \
            "${DISCOVER_ARRAY_SLURM_SCRIPT}")
        echo "${DISCOVER_ARRAY_SUBMIT_OUTPUT}"
        DISCOVER_ARRAY_JOB_ID=$(echo "${DISCOVER_ARRAY_SUBMIT_OUTPUT}" | awk '{print $NF}')
        if [[ -z "${DISCOVER_ARRAY_JOB_ID}" ]]; then
            echo "Error: could not parse discovery array job id from: ${DISCOVER_ARRAY_SUBMIT_OUTPUT}"
            exit 1
        fi

        DISCOVER_MERGE_SUBMIT_OUTPUT=$(sbatch \
            --dependency="afterok:${DISCOVER_ARRAY_JOB_ID}" \
            --output="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/discover_merge_%A.out" \
            --error="${SCRATCH_ROOT}/logs/logs_%A_${DATETIME}/discover_merge_%A.err" \
            --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},DISCOVER_DUMP_LIST=${DISCOVER_DUMP_LIST},DISCOVER_OUT_DIR=${DISCOVER_OUT_DIR}" \
            "${DISCOVER_MERGE_SLURM_SCRIPT}")
        echo "${DISCOVER_MERGE_SUBMIT_OUTPUT}"
        DISCOVER_MERGE_JOB_ID=$(echo "${DISCOVER_MERGE_SUBMIT_OUTPUT}" | awk '{print $NF}')
        if [[ -z "${DISCOVER_MERGE_JOB_ID}" ]]; then
            echo "Error: could not parse discovery merge job id from: ${DISCOVER_MERGE_SUBMIT_OUTPUT}"
            exit 1
        fi
    fi
    step_done
fi

step_start "Submitting tracing job"
if [[ ! -f "${TRACING_SLURM_SCRIPT}" ]]; then
    echo "Error: tracing slurm script not found: ${TRACING_SLURM_SCRIPT}"
    exit 1
fi

TRACING_DEPENDENCY_ARGS=()
if [[ -n "${DISCOVER_MERGE_JOB_ID}" ]]; then
    TRACING_DEPENDENCY_ARGS=(--dependency="afterok:${DISCOVER_MERGE_JOB_ID}")
fi

TRACING_SUBMIT_OUTPUT=$(sbatch \
    "${TRACING_DEPENDENCY_ARGS[@]}" \
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
step_done

step_start "Submitting run_batch array job(s)"
ARRAY_JOB_IDS=()
ARRAY_JOB_ID=""  # first chunk job id, used for LOG_DIR name

# Submit chunks in groups of CHUNKS_PER_WAVE simultaneously. Each wave depends
# on the last chunk of the previous wave so only one wave is in the queue at a
# time, keeping total submitted jobs within the QOS MaxSubmitPU limit.
# Overhead from tracing + merge array + summary = ~1603 jobs, leaving ~8397
# slots for batch tasks. At MaxArraySize=2048, CHUNKS_PER_WAVE=4 uses 4x2048
# =8192 slots safely. The final wave may be smaller.
CHUNKS_PER_WAVE=4
PREV_WAVE_LAST_JOB_ID=""

CHUNK_COUNT="${#ARRAY_CHUNK_OFFSETS[@]}"
i=0
while [[ "${i}" -lt "${CHUNK_COUNT}" ]]; do
    # Collect the indices for this wave
    WAVE_JOB_IDS=()
    WAVE_END=$(( i + CHUNKS_PER_WAVE ))
    if [[ "${WAVE_END}" -gt "${CHUNK_COUNT}" ]]; then
        WAVE_END="${CHUNK_COUNT}"
    fi

    for (( j=i; j<WAVE_END; j++ )); do
        CHUNK_OFFSET="${ARRAY_CHUNK_OFFSETS[$j]}"
        CHUNK_SIZE="${ARRAY_CHUNK_SIZES[$j]}"
        CHUNK_ARRAY_MAX=$(( CHUNK_SIZE - 1 ))
        CHUNK_ARRAY_SPEC="0-${CHUNK_ARRAY_MAX}"
        if [[ -n "${MAX_CONCURRENT}" ]]; then
            CHUNK_ARRAY_SPEC="${CHUNK_ARRAY_SPEC}%${MAX_CONCURRENT}"
        fi

        # First wave depends on tracing; subsequent waves depend on the last
        # chunk of the previous wave (afterany so failed batches don't block).
        if [[ -z "${PREV_WAVE_LAST_JOB_ID}" ]]; then
            CHUNK_DEPENDENCY="afterok:${TRACING_JOB_ID}"
        else
            CHUNK_DEPENDENCY="afterany:${PREV_WAVE_LAST_JOB_ID}"
        fi

        CHUNK_SUBMIT_OUTPUT=$(sbatch \
            --dependency="${CHUNK_DEPENDENCY}" \
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
    done

    # Next wave depends on the last chunk of this wave
    PREV_WAVE_LAST_JOB_ID="${WAVE_JOB_IDS[-1]}"
    i="${WAVE_END}"
done

LOG_DIR="${SCRATCH_ROOT}/logs/logs_${ARRAY_JOB_ID}_${DATETIME}"
mkdir -p "${LOG_DIR}"

echo "Run-batch chunk job ids : ${ARRAY_JOB_IDS[*]}"
echo "Batch logs              : ${LOG_DIR}"
echo "Tracing logs            : ${TRACE_LOG_DIR}"
step_done

step_start "Submitting merge_dumps array job"
if [[ ! -f "${MERGE_SLURM_SCRIPT}" ]]; then
    echo "Error: merge slurm script not found: ${MERGE_SLURM_SCRIPT}"
    exit 1
fi

ALL_BATCH_JOB_IDS=$(IFS=:; echo "${ARRAY_JOB_IDS[*]}")

MERGE_SUBMIT_OUTPUT=$(sbatch \
    --dependency="afterany:${ALL_BATCH_JOB_IDS}" \
    --array="${MERGE_ARRAY_SPEC}" \
    --output="${LOG_DIR}/merge_%A_%a.out" \
    --error="${LOG_DIR}/merge_%A_%a.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN}" \
    "${MERGE_SLURM_SCRIPT}")
echo "${MERGE_SUBMIT_OUTPUT}"

MERGE_JOB_ID=$(echo "${MERGE_SUBMIT_OUTPUT}" | awk '{print $NF}')
if [[ -z "${MERGE_JOB_ID}" ]]; then
    echo "Error: could not parse merge job id from: ${MERGE_SUBMIT_OUTPUT}"
    exit 1
fi
step_done

step_start "Submitting post-run summary job"
if [[ ! -f "${SUMMARY_SLURM_SCRIPT}" ]]; then
    echo "Error: summary slurm script not found: ${SUMMARY_SLURM_SCRIPT}"
    exit 1
fi

SUMMARY_SUBMIT_OUTPUT=$(sbatch \
    --dependency="afterany:${MERGE_JOB_ID}" \
    --output="${LOG_DIR}/summary_%A.out" \
    --error="${LOG_DIR}/summary_%A.err" \
    --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},BATCH_LOG_DIR=${LOG_DIR},BATCH_ARRAY_JOB_ID=${ARRAY_JOB_ID}" \
    "${SUMMARY_SLURM_SCRIPT}")
echo "${SUMMARY_SUBMIT_OUTPUT}"
step_done

echo "Batch submission workflow completed."
echo "Submitted chain: ${DISCOVER_ARRAY_JOB_ID:+discover_array -> }${DISCOVER_MERGE_JOB_ID:+discover_merge -> }tracing -> run_batch_array (${#ARRAY_JOB_IDS[@]} chunk(s)) -> merge_dumps_array -> summarize"
echo "Tracing job id          : ${TRACING_JOB_ID}"
echo "Run-batch chunk job ids : ${ARRAY_JOB_IDS[*]}"
echo "Merge array job id      : ${MERGE_JOB_ID}"
echo "Logs: ${LOG_DIR}"