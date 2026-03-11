#!/bin/bash
#SBATCH --job-name=trace_parallel
#SBATCH --output=logs/logs_%A/trace_%A_%a.out  # overridden by submit_trace_parallel.sh
#SBATCH --error=logs/logs_%A/trace_%A_%a.err   # overridden by submit_trace_parallel.sh
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-0  # overridden dynamically by submit_trace_parallel.sh

# Runs phantomanalysis on one pair of dumps separated by STEP.
# The first dump is used for initialisation only (no output).
# The second dump produces one line per tracked particle in trace_output/*.phys.
#
# NOTE: All concurrent jobs append to the same particle .phys files.
#       Lines from different jobs may arrive out of time order.
#       Sort each .phys file by the first column (time) after all jobs finish.

set -euo pipefail

WORK_DIR="${WORK_DIR:-/fred/oz304/beckers/pigru}"
DATA_DIR="${DATA_DIR:-/fred/oz304/tdanilov/pigru}"
PREFIX="${PREFIX:-pigru}"
STEP="${STEP:-10}"

# Read the first dump of this task's pair from the list written by submit_trace_parallel.sh
PAIR_LIST="${WORK_DIR}/trace_pair_list.txt"
FIRST=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PAIR_LIST")

if [[ -z "$FIRST" ]]; then
    echo "Error: no entry for task ${SLURM_ARRAY_TASK_ID} in ${PAIR_LIST}"
    exit 1
fi

SECOND=$(( FIRST + STEP ))
DUMP1="${DATA_DIR}/${PREFIX}_$(printf '%05d' "${FIRST}")"
DUMP2="${DATA_DIR}/${PREFIX}_$(printf '%05d' "${SECOND}")"

for D in "$DUMP1" "$DUMP2"; do
    if [[ ! -f "$D" ]]; then
        echo "Error: dump file not found: $D"
        exit 1
    fi
done

echo "Starting job ${SLURM_ARRAY_JOB_ID:-NA}, array task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
echo "Processing pair: $(basename "$DUMP1")  $(basename "$DUMP2")"

cd "$WORK_DIR"
mkdir -p trace_output

./phantomanalysis "$DUMP1" "$DUMP2"

# Mark the output dump as done so submit_trace_parallel.sh can skip it on re-run
touch "trace_output/.done_$(printf '%05d' "${SECOND}")"

echo "Pair (${FIRST}, ${SECOND}) complete."
