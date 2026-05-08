#!/bin/bash
# Scatter/gather merge submitter for OzSTAR
#
# Features:
#   - 2048-array chunking
#   - scatter-only mode
#   - gather-only mode
#   - resume mode
#   - dependency chaining
#   - wave submission compatible with OzSTAR limits
#
# Examples:
#
# First scatter run:
#   ./workflow/submit_scatter_gather.sh --scatter-only
#
# First gather run:
#   ./workflow/submit_scatter_gather.sh --gather-only
#
# Resume scatter:
#   ./workflow/submit_scatter_gather.sh \
#       --scatter-only \
#       --resume-scatter scatter_status.txt
#
# Resume gather:
#   ./workflow/submit_scatter_gather.sh \
#       --gather-only \
#       --resume-gather gather_status.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_CONFIG="${PIPELINE_ROOT}/config/pipeline_config.yaml"

SCATTER_SLURM="${PIPELINE_ROOT}/slurm/scatter_batches.slurm"
GATHER_SLURM="${PIPELINE_ROOT}/slurm/gather_dumps.slurm"

MAX_ARRAY_SIZE=2048

CONFIG_PATH="${DEFAULT_CONFIG}"

RESUME_SCATTER_FILE=""
RESUME_GATHER_FILE=""
AFTER_JOB_ID=""
EXTERNAL_LOG_DIR=""

SCATTER_ONLY=0
GATHER_ONLY=0

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

        --log-dir)
            EXTERNAL_LOG_DIR="$2"
            shift 2
            ;;

        --resume-scatter)
            RESUME_SCATTER_FILE="$2"
            shift 2
            ;;

        --resume-gather)
            RESUME_GATHER_FILE="$2"
            shift 2
            ;;

        --scatter-only)
            SCATTER_ONLY=1
            shift
            ;;

        --gather-only)
            GATHER_ONLY=1
            shift
            ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;

    esac

done

if [[ "${SCATTER_ONLY}" -eq 1 && "${GATHER_ONLY}" -eq 1 ]]; then
    echo "Cannot use both --scatter-only and --gather-only"
    exit 1
fi

readarray -t INFO < <(
    /fred/oz304/beckers/MRP_env/bin/python - <<'PY' "${CONFIG_PATH}" "${PIPELINE_ROOT}"
import sys
from pathlib import Path

config_path   = Path(sys.argv[1]).resolve()
pipeline_root = Path(sys.argv[2]).resolve()

sys.path.insert(0, str(pipeline_root / "scripts"))

from common import DumpSelection, PipelineConfigManager

cfg = PipelineConfigManager.load(config_path)

python_bin = cfg.get(
    "paths",
    {}
).get(
    "python",
    "/fred/oz304/beckers/MRP_env/bin/python"
)

scratch_root = cfg.get(
    "paths",
    {}
).get(
    "scratch_root",
    str(pipeline_root / "work")
)

dump_count = len(
    DumpSelection.target_dump_numbers(cfg)
)

batch_dir = Path(
    cfg["paths"]["batch_output_dir"]
)

batch_count = len(
    sorted(batch_dir.glob("batch_*.h5"))
)

print(python_bin)
print(scratch_root)
print(batch_count)
print(dump_count)
PY
)

PYTHON_BIN="${INFO[0]}"
SCRATCH_ROOT="${INFO[1]}"
TOTAL_BATCH_COUNT="${INFO[2]}"
TOTAL_DUMP_COUNT="${INFO[3]}"

mkdir -p "${SCRATCH_ROOT}/logs"

DATETIME=$(date +%Y-%m-%d_%H-%M-%S)

if [[ -n "${EXTERNAL_LOG_DIR}" ]]; then
    LOG_DIR="${EXTERNAL_LOG_DIR}"
else
    LOG_DIR="${SCRATCH_ROOT}/logs/logs_${DATETIME}"
fi

mkdir -p "${LOG_DIR}"

SCATTER_PENDING_FILE="${SCRATCH_ROOT}/logs/pending_scatter_${DATETIME}.txt"
GATHER_PENDING_FILE="${SCRATCH_ROOT}/logs/pending_gather_${DATETIME}.txt"

build_pending_file() {

    local total_count="$1"
    local resume_file="$2"
    local out_file="$3"

    if [[ -z "${resume_file}" ]]; then

        seq 0 $((total_count - 1)) > "${out_file}"

    else

        /fred/oz304/beckers/MRP_env/bin/python - \
            "${resume_file}" \
            "${total_count}" \
            "${out_file}" <<'PY'

import sys
from pathlib import Path

status_file = Path(sys.argv[1])
total_count = int(sys.argv[2])
out_file    = Path(sys.argv[3])

completed = set()

for line in status_file.read_text().splitlines():

    line = line.strip()

    if not line:
        continue

    if line.startswith("#"):
        continue

    idx, status = line.split()[:2]

    if status.upper() == "COMPLETED":
        completed.add(int(idx))

pending = [
    i
    for i in range(total_count)
    if i not in completed
]

out_file.write_text(
    "\n".join(map(str, pending)) + "\n"
)

print(len(pending))
PY

    fi

}

build_pending_file \
    "${TOTAL_BATCH_COUNT}" \
    "${RESUME_SCATTER_FILE}" \
    "${SCATTER_PENDING_FILE}"

build_pending_file \
    "${TOTAL_DUMP_COUNT}" \
    "${RESUME_GATHER_FILE}" \
    "${GATHER_PENDING_FILE}"

SCATTER_COUNT=$(wc -l < "${SCATTER_PENDING_FILE}" | tr -d ' ')
GATHER_COUNT=$(wc -l < "${GATHER_PENDING_FILE}" | tr -d ' ')

echo "========================================="
echo "Scatter/Gather Submission"
echo "========================================="
echo "Scatter tasks : ${SCATTER_COUNT}/${TOTAL_BATCH_COUNT}"
echo "Gather tasks  : ${GATHER_COUNT}/${TOTAL_DUMP_COUNT}"
echo "Log directory : ${LOG_DIR}"
echo "========================================="

submit_chunks() {

    local task_count="$1"
    local pending_file="$2"
    local slurm_file="$3"
    local stage_name="$4"
    local dependency="$5"

    local submitted_jobs=()

    local start=0

    while [[ "${start}" -lt "${task_count}" ]]; do

        local end=$(( start + MAX_ARRAY_SIZE - 1 ))

        if [[ "${end}" -ge "${task_count}" ]]; then
            end=$(( task_count - 1 ))
        fi

        local chunk_size=$(( end - start + 1 ))

        local dep_args=()

        if [[ -n "${dependency}" ]]; then
            dep_args+=(--dependency="afterany:${dependency}")
        fi

        OUT=$(sbatch \
            "${dep_args[@]}" \
            --array="0-$((chunk_size - 1))" \
            --output="${LOG_DIR}/${stage_name}_%A_%a.out" \
            --error="${LOG_DIR}/${stage_name}_%A_%a.err" \
            --export="PIPELINE_ROOT=${PIPELINE_ROOT},PIPELINE_CONFIG=${CONFIG_PATH},PIPELINE_PYTHON=${PYTHON_BIN},PENDING_INDEX_FILE=${pending_file},INDEX_OFFSET=${start}" \
            "${slurm_file}"
        )

        JOB_ID=$(echo "${OUT}" | awk '{print $NF}')

        echo "${OUT}"

        submitted_jobs+=("${JOB_ID}")

        start=$(( end + 1 ))

    done

    echo "$(IFS=:; echo "${submitted_jobs[*]}")"

}

SCATTER_JOB_IDS=""

if [[ "${GATHER_ONLY}" -eq 0 ]]; then

    echo
    echo "Submitting scatter jobs..."
    echo

    SCATTER_JOB_IDS=$(
        submit_chunks \
            "${SCATTER_COUNT}" \
            "${SCATTER_PENDING_FILE}" \
            "${SCATTER_SLURM}" \
            "scatter" \
            "${AFTER_JOB_ID}"
    )

    echo
    echo "Scatter jobs:"
    echo "${SCATTER_JOB_IDS}"

fi

if [[ "${SCATTER_ONLY}" -eq 0 ]]; then

    echo
    echo "Submitting gather jobs..."
    echo

    GATHER_DEP=""

    if [[ -n "${SCATTER_JOB_IDS}" ]]; then
        GATHER_DEP="${SCATTER_JOB_IDS}"
    else
        GATHER_DEP="${AFTER_JOB_ID}"
    fi

    GATHER_JOB_IDS=$(
        submit_chunks \
            "${GATHER_COUNT}" \
            "${GATHER_PENDING_FILE}" \
            "${GATHER_SLURM}" \
            "gather" \
            "${GATHER_DEP}"
    )

    echo
    echo "Gather jobs:"
    echo "${GATHER_JOB_IDS}"

fi

echo
echo "Done."