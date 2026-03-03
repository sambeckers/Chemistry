#!/bin/bash
# Submit run_em_parallel.slurm with the array size set automatically
# to match the number of selected particle IDs in run_evolving_models.py.
#
# Usage:
#   ./submit_em_parallel.sh Crich
#   ./submit_em_parallel.sh Orich
#   ./submit_em_parallel.sh Crich analysis

set -euo pipefail

CHEMISTRY_ARG="${1:-}"
ANALYSIS_ARG="${2:-}"

if [[ -z "$CHEMISTRY_ARG" ]]; then
    echo "Error: chemistry type is required."
    echo "Usage: ./submit_em_parallel.sh Crich|Orich [analysis]"
    exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/fred/oz304/beckers/MRP_env/bin/python"
SCRIPT="$WORKDIR/run_evolving_models.py"
SLURM="$WORKDIR/run_em_parallel.slurm"

# Query the particle count without running the models
N=$("$PYTHON" "$SCRIPT" --count --chemistry-type "$CHEMISTRY_ARG")

if [[ -z "$N" || "$N" -lt 1 ]]; then
    echo "Error: got invalid particle count '$N' from --count."
    exit 1
fi

ARRAY_MAX=$(( N - 1 ))
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="logs/logs_%A_${DATETIME}"

echo "Submitting array job: $N particle(s), --array=0-${ARRAY_MAX}, logs -> ${LOG_DIR}"

sbatch \
    --array="0-${ARRAY_MAX}" \
    --output="${LOG_DIR}/em_parallel_%A_%a.out" \
    --error="${LOG_DIR}/em_parallel_%A_%a.err" \
    "$SLURM" "$CHEMISTRY_ARG" ${ANALYSIS_ARG:+"$ANALYSIS_ARG"}
