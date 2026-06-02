#!/bin/bash
# =============================================================================
# animate_renders.slurm
#
# Embarrassingly parallel frame generation for animate_renders.py.
# Each array task renders a 1/N_TASKS subset of all dumps.
#
# Typical submission (adjust --array and --export to taste):
#
#   # Fixed xlim, CO in xz plane, 32 tasks
#   sbatch --export=ALL,QUANTITY=CO,PLANE=xz,XLIM_MODE=fixed,XLIM=1000 \
#          animate_renders.slurm
#
#   # Ramping xlim, 64 tasks
#   sbatch --export=ALL,QUANTITY=HCN,PLANE=xy,\
#          XLIM_MODE=ramp,XLIM_START=100,XLIM_END=2000 \
#          --array=0-63 \
#          animate_renders.slurm
#
# After all jobs finish, assemble with:
#   python animate_renders.py assemble \
#       --quantity CO --plane xz --xlim-mode fixed --xlim 1000 --fps 15
# =============================================================================

#SBATCH --job-name=anim_frames
#SBATCH --output=logs_%A/anim_%A_%a.out
#SBATCH --error=logs_%A/anim_%A_%a.err
#SBATCH --array=0-31           # 32 tasks — increase if you have many dumps
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # one render per task; raise if sarracen is threaded
#SBATCH --mem=16G              # SPH rendering can be memory-hungry
#SBATCH --time=02:00:00        # adjust based on dump size and number of frames

# ---------------------------------------------------------------------------
# Environment variables with defaults (override via --export on sbatch)
# ---------------------------------------------------------------------------
QUANTITY=${QUANTITY:-density}
PLANE=${PLANE:-xy}
XLIM_MODE=${XLIM_MODE:-fixed}
XLIM=${XLIM:-1000}
XLIM_START=${XLIM_START:-100}
XLIM_END=${XLIM_END:-2000}
CHEMISTRY=${CHEMISTRY:-Crich}
XSEC=${XSEC:-}          # leave empty for column-integrated render
COL_DENS=${COL_DENS:-0} # set to 1 to enable column-density weighting
CMAP=${CMAP:-gist_heat}

# Number of SLURM tasks derived from the array range
N_TASKS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

echo "======================================================="
echo "  Array task   : ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}"
echo "  Quantity      : ${QUANTITY}"
echo "  Plane         : ${PLANE}"
echo "  xlim mode     : ${XLIM_MODE}  (fixed=${XLIM}, ramp=${XLIM_START}→${XLIM_END})"
echo "  Chemistry     : ${CHEMISTRY}"
echo "======================================================="

# Create log directory (harmless if it already exists)
mkdir -p "logs_${SLURM_ARRAY_JOB_ID}"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

# ---------------------------------------------------------------------------
# Paths — adjust SCRIPT_DIR if animate_renders.py lives elsewhere
# ---------------------------------------------------------------------------
SCRIPT_DIR="/fred/oz304/beckers/Chemistry/pipeline/plotting"

# ---------------------------------------------------------------------------
# Build optional flags
# ---------------------------------------------------------------------------
XSEC_FLAG=""
if [ -n "${XSEC}" ]; then
    XSEC_FLAG="--xsec ${XSEC}"
fi

COL_DENS_FLAG=""
if [ "${COL_DENS}" = "1" ]; then
    COL_DENS_FLAG="--col-dens"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${SCRIPT_DIR}/animate_renders.py" frames \
    --quantity    "${QUANTITY}"              \
    --plane       "${PLANE}"                \
    --xlim-mode   "${XLIM_MODE}"            \
    --xlim        "${XLIM}"                 \
    --xlim-start  "${XLIM_START}"           \
    --xlim-end    "${XLIM_END}"             \
    --chemistry   "${CHEMISTRY}"            \
    --cmap        "${CMAP}"                 \
    ${XSEC_FLAG}                            \
    ${COL_DENS_FLAG}                        \
    --task-id     "${SLURM_ARRAY_TASK_ID}"  \
    --n-tasks     "${N_TASKS}"
