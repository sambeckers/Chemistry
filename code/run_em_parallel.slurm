#!/bin/bash
#SBATCH --job-name=em_parallel
#SBATCH --output=logs/logs_%A/em_parallel_%A_%a.out
#SBATCH --error=logs/logs_%A/em_parallel_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-14

set -euo pipefail

WORKDIR="/fred/oz304/beckers/Chemistry/code"
cd "$WORKDIR"

CHEMISTRY_ARG="${1:-}"
if [[ -n "$CHEMISTRY_ARG" ]]; then
	export CHEMISTRY_TYPE="$CHEMISTRY_ARG"
fi

ANALYSIS_ARG="${2:-}"
ANALYSIS_FLAG="--no-analysis"
if [[ -n "$ANALYSIS_ARG" ]]; then
	if [[ "$ANALYSIS_ARG" == "analysis" ]]; then
		ANALYSIS_FLAG="--analysis"
	else
		echo "Error: optional second argument must be 'analysis' if provided."
		echo "Usage: sbatch run_em_parallel.slurm Crich [analysis]"
		exit 1
	fi
fi

if [[ -z "${CHEMISTRY_TYPE:-}" ]]; then
	echo "Error: chemistry type is required."
	echo "Usage: sbatch run_em_parallel.slurm Crich"
	echo "   or: sbatch run_em_parallel.slurm Orich"
	echo "   or: CHEMISTRY_TYPE=Crich sbatch run_em_parallel.slurm"
	exit 1
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/MRP_env

echo "Starting job ${SLURM_ARRAY_JOB_ID:-NA}, array task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
echo "Chemistry type: ${CHEMISTRY_TYPE:-<default set in python script>}"
echo "Analysis mode: $([[ "$ANALYSIS_FLAG" == "--analysis" ]] && echo true || echo false)"

# --slurm enables non-interactive mode and single-particle selection via
# SLURM_ARRAY_TASK_ID.  Science settings (paths, particle selection, analysis
# flags, …) are configured inside run_chemistry_models.py — keep --array
# in sync with n_select defined there.
python run_evolving_models.py --parallel --chemistry-type "$CHEMISTRY_TYPE" "$ANALYSIS_FLAG"
