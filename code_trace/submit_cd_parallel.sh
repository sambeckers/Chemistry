#!/bin/bash
# Submit column density calculations as a SLURM array job.
# One CPU per dump, only unprocessed dumps are queued.
#
# Usage:
#   ./process_all_dumps.sh
#   ./process_all_dumps.sh --start 100 --end 500
#   ./process_all_dumps.sh --step 10          # every 10th dump (0, 10, 20, ...)
#   ./process_all_dumps.sh --prefix wind      # for wind_NNNNN dumps

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR="/fred/oz304/tdanilov/pigru"  # Phantom dump directory
WORK_DIR="/fred/oz304/beckers/pigru"   # Working dir (AV/, PhotoData/, Models/ live here)
PREFIX="pigru"                         # Dump file prefix (e.g. pigru or wind)
START_DUMP=0
END_DUMP=711
STEP=10                                # Process every Nth dump (1 = all, 10 = every 10th)

# ── Parse optional arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)  START_DUMP="$2"; shift 2 ;;
        --end)    END_DUMP="$2";   shift 2 ;;
        --step)   STEP="$2";       shift 2 ;;
        --prefix) PREFIX="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SLURM="$SCRIPT_DIR/run_cd_parallel.slurm"

# ── Find unprocessed dumps ────────────────────────────────────────────────────
UNPROCESSED=()

for DUMP_FILE in $(ls "${DATA_DIR}"/${PREFIX}_* 2>/dev/null | grep -E "${PREFIX}_[0-9]{5}$" | sort -V); do
    DUMP_NUM=$(basename "${DUMP_FILE}" | sed "s/${PREFIX}_//" | sed 's/^0*//' | sed 's/^$/0/')
    [[ "${DUMP_NUM}" -lt "${START_DUMP}" || "${DUMP_NUM}" -gt "${END_DUMP}" ]] && continue
    [[ $(( DUMP_NUM % STEP )) -ne 0 ]] && continue
    AV_FILE="${WORK_DIR}/AV/AV_$(printf '%05d' "${DUMP_NUM}")"
    if [[ ! -f "${AV_FILE}" ]]; then
        UNPROCESSED+=("${DUMP_NUM}")
    fi
done

N=${#UNPROCESSED[@]}

if [[ "${N}" -eq 0 ]]; then
    echo "All dumps already processed. Nothing to submit."
    exit 0
fi

# Write the dump list so the slurm script can index into it via SLURM_ARRAY_TASK_ID
DUMP_LIST="${WORK_DIR}/dump_list.txt"
printf '%s\n' "${UNPROCESSED[@]}" > "${DUMP_LIST}"

ARRAY_MAX=$(( N - 1 ))
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="${SCRIPT_DIR}/logs/logs_%A_${DATETIME}"

echo "========================================="
echo "Column Density Calculation (SLURM)"
echo "========================================="
echo "Data directory : ${DATA_DIR}"
echo "Work directory : ${WORK_DIR}"
echo "Prefix         : ${PREFIX}"
echo "Step           : ${STEP}"
echo "Dumps to submit: ${N}  (array 0-${ARRAY_MAX})"
echo "Dump list      : ${DUMP_LIST}"
echo "Logs           : ${LOG_DIR}"
echo ""

sbatch \
    --array="0-${ARRAY_MAX}" \
    --output="${LOG_DIR}/cd_parallel_%A_%a.out" \
    --error="${LOG_DIR}/cd_parallel_%A_%a.err" \
    --export="DATA_DIR=${DATA_DIR},WORK_DIR=${WORK_DIR},PREFIX=${PREFIX}" \
    "${SLURM}"
