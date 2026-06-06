#!/bin/bash
# =============================================================================
# molecule_art.slurm
#
# Example SLURM array job for the molecule grid art.
# Each array task renders a subset of species. After all tasks finish,
# run the assembly step (see below).
#
# Usage:
#   sbatch --array=0-19 --export=ALL,DUMP_INDEX=1581 molecule_art.slurm
# =============================================================================
#SBATCH --job-name=mol_art
#SBATCH --output=logs_%A/molecule_art_%A_%a.out
#SBATCH --error=logs_%A/molecule_art_%A_%a.err
#SBATCH --array=0-49            # 20 tasks — adjust based on species count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00

DUMP_INDEX=${DUMP_INDEX:-1581}
CHEMISTRY=${CHEMISTRY:-Crich}
PLANE=${PLANE:-xy}
XLIM=${XLIM:-100}
CMAP=${CMAP:-gist_heat}

N_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

echo "---------------------------------------------------------"
echo "  Array task   : ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}"
echo "  Dump index   : ${DUMP_INDEX}"
echo "  Plane / xlim : ${PLANE} / ${XLIM}"
echo "---------------------------------------------------------"

mkdir -p "logs_${SLURM_ARRAY_JOB_ID}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

SCRIPT="/fred/oz304/beckers/Chemistry/pipeline/plotting/molecule_grid_art.py"
OUTDIR="/fred/oz304/beckers/Chemistry/figures/v10a09_out/molecule_art/panels"

python "${SCRIPT}" panels \
    --dump-index   "${DUMP_INDEX}" \
    --dump-dir     "/fred/oz304/beckers/v10a09_out/output/dumps" \
    --phantom-dir  "/fred/oz304/beckers/v10a09" \
    --chemistry    "${CHEMISTRY}" \
    --plane        "${PLANE}" \
    --xlim         "${XLIM}" \
    --cmap         "${CMAP}" \
    --output-dir   "${OUTDIR}" \
    --task-id      "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks      "${N_TASKS}"