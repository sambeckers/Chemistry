#!/bin/bash
# =============================================================================
# submit_frames.slurm
#
# Render animation frames in parallel as a SLURM array job.
#
# USAGE
# -----
# Minimal (uses animate_config.yaml in the submit directory):
#   sbatch submit_frames.slurm
#
# With overrides (absolute path recommended for CONFIG):
#   sbatch --export=ALL,CONFIG=/abs/path/to/animate_config.yaml,MODE=single,QUANTITY=CO \
#          submit_frames.slurm
#
# IMPORTANT: Keep N_TASKS below in sync with --array.
#   --array=0-31  →  N_TASKS=32
#   --array=0-15  →  N_TASKS=16
# Do NOT derive N_TASKS from SLURM_ARRAY_TASK_{MIN,MAX}; that formula breaks
# for stepped arrays (e.g. --array=0-31:2).
#
# After all frames finish, submit the assembly job with a dependency:
#   FRAMES_JID=$(sbatch --parsable submit_frames.slurm)
#   sbatch --dependency=afterok:$FRAMES_JID submit_assemble.slurm
# =============================================================================
#SBATCH --job-name=anim_frames
#SBATCH --output=logs/logs_%A/anim_%A_%a.out
#SBATCH --error=logs/logs_%A/anim_%A_%a.err
#SBATCH --array=0-255
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G          # increase to 16G for mol_grid
#SBATCH --time=00:30:00           # increase to 01:30:00 for mol_grid

# ---------------------------------------------------------------------------
# Configuration — CONFIG must be an absolute path.
# Default: animate_config.yaml in the directory from which sbatch was called.
# ---------------------------------------------------------------------------
CONFIG=${CONFIG:-"${SLURM_SUBMIT_DIR}/animate_config.yaml"}

# Number of parallel tasks — must equal the number of slots in --array above.
N_TASKS=256

mkdir -p "logs/logs_${SLURM_ARRAY_JOB_ID}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

SCRIPT=/fred/oz304/beckers/Chemistry/pipeline/plotting/animate_renders_v2.py

# ---------------------------------------------------------------------------
# Build the argument list.
#
# Pattern for optional string / numeric overrides:
#   ${VAR:+--flag "$VAR"}   — emits --flag VALUE only when VAR is non-empty.
#
# Pattern for store_true (boolean) flags:
#   ${VAR:+--flag}          — emits --flag (no value) only when VAR is non-empty.
#   These flags must NOT receive a value; passing "" crashes argparse.
#
# Set overrides at submission time via --export=ALL,VAR=value.
# ---------------------------------------------------------------------------
python "$SCRIPT" frames \
  --config        "$CONFIG" \
  ${MODE:+--mode          "$MODE"} \
  ${PLANE:+--plane        "$PLANE"} \
  ${CHEMISTRY:+--chemistry "$CHEMISTRY"} \
  ${XLIM_MODE:+--xlim-mode "$XLIM_MODE"} \
  ${XLIM:+--xlim          "$XLIM"} \
  ${XLIM_START:+--xlim-start "$XLIM_START"} \
  ${XLIM_END:+--xlim-end  "$XLIM_END"} \
  ${XSEC:+--xsec          "$XSEC"} \
  ${QUANTITY:+--quantity   $QUANTITY} \
  ${CMAP:+--cmap          "$CMAP"} \
  ${FPS:+--fps            "$FPS"} \
  ${DARK_MODE:+--dark-mode} \
  ${COL_DENS:+--col-dens} \
  ${DENS_WEIGHT:+--dens-weight} \
  --task-id  "$SLURM_ARRAY_TASK_ID" \
  --n-tasks  "$N_TASKS"
