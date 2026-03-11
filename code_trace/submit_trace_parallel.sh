#!/bin/bash
# Submit phantomanalysis as a SLURM array job.
# Each task processes one pair of dumps separated by STEP:
#   ./phantomanalysis PREFIX_N PREFIX_{N+STEP}
# The first dump is used for initialisation only (no .phys output).
# The second dump writes one line per tracked particle.
#
# Usage:
#   ./submit_trace_parallel.sh
#   ./submit_trace_parallel.sh --start 100 --end 500
#   ./submit_trace_parallel.sh --step 10          # pairs (0,10),(10,20),(20,30),...
#   ./submit_trace_parallel.sh --prefix wind      # for wind_NNNNN dumps
#   ./submit_trace_parallel.sh --max-concurrent 64
#   (no limit on concurrent tasks by default)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR="/fred/oz304/tdanilov/pigru"  # Phantom dump directory
WORK_DIR="/fred/oz304/beckers/pigru_out"   # Working dir (phantomanalysis binary, trace.cfg, trace_output/)
PREFIX="pigru"                         # Dump file prefix
START_DUMP=0
END_DUMP=700          # Last first-of-pair (second will be END_DUMP+STEP)
STEP=10               # Gap between the two dumps in each pair

# ── Parse optional arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)          START_DUMP="$2";     shift 2 ;;
        --end)            END_DUMP="$2";       shift 2 ;;
        --step)           STEP="$2";           shift 2 ;;
        --prefix)         PREFIX="$2";         shift 2 ;;

        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="/fred/oz304/beckers/Chemistry/code_trace"
SLURM="$SCRIPT_DIR/run_trace_parallel.slurm"

# ── Find unprocessed pairs ─────────────────────────────────────────────────────
UNPROCESSED=()

if (( STEP <= 0 )); then
    echo "Error: --step must be a positive integer"
    exit 1
fi

for (( N=START_DUMP; N<=END_DUMP; N+=STEP )); do
    SECOND=$(( N + STEP ))
    DUMP1="${DATA_DIR}/${PREFIX}_$(printf '%05d' "${N}")"
    DUMP2="${DATA_DIR}/${PREFIX}_$(printf '%05d' "${SECOND}")"

    # Both files must exist
    [[ ! -f "$DUMP1" ]] && continue
    [[ ! -f "$DUMP2" ]] && continue

    SENTINEL="${WORK_DIR}/trace_output/.done_$(printf '%05d' "${SECOND}")"
    if [[ ! -f "$SENTINEL" ]]; then
        UNPROCESSED+=("${N}")
    fi
done

N_JOBS=${#UNPROCESSED[@]}

if [[ "${N_JOBS}" -eq 0 ]]; then
    echo "All pairs already processed. Nothing to submit."
    exit 0
fi

# Write the pair list so the slurm script can index into it via SLURM_ARRAY_TASK_ID
PAIR_LIST="${WORK_DIR}/trace_pair_list.txt"
printf '%s\n' "${UNPROCESSED[@]}" > "${PAIR_LIST}"

ARRAY_MAX=$(( N_JOBS - 1 ))
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="${SCRIPT_DIR}/logs/logs_%A_${DATETIME}"

echo "========================================="
echo "Phantom Trace Analysis (SLURM)"
echo "========================================="
echo "Data directory   : ${DATA_DIR}"
echo "Work directory   : ${WORK_DIR}"
echo "Prefix           : ${PREFIX}"
echo "Step             : ${STEP}"
echo "Pairs to submit  : ${N_JOBS}  (array 0-${ARRAY_MAX})"
echo "Pair list        : ${PAIR_LIST}"
echo "Logs             : ${LOG_DIR}"
echo ""
echo "NOTE: All jobs append to the same trace_output/*.phys files."
echo "      Sort each file by the time column after completion if needed."
echo ""

sbatch \
    --array="0-${ARRAY_MAX}" \
    --output="${LOG_DIR}/trace_%A_%a.out" \
    --error="${LOG_DIR}/trace_%A_%a.err" \
    --export="DATA_DIR=${DATA_DIR},WORK_DIR=${WORK_DIR},PREFIX=${PREFIX},STEP=${STEP}" \
    "${SLURM}"
