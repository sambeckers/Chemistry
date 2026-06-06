#!/bin/bash
# =============================================================================
# submit_assemble.slurm
#
# Stitch pre-rendered PNG frames into an MP4 via matplotloom.
# Submit this after all frame jobs have completed.
#
# USAGE
# -----
# Standalone (positional config argument):
#   sbatch submit_assemble.slurm /abs/path/to/animate_config.yaml
#
# As a dependency on the frame array job (recommended):
#   FRAMES_JID=$(sbatch --parsable submit_frames.slurm)
#   sbatch --dependency=afterok:$FRAMES_JID \
#          submit_assemble.slurm /abs/path/to/animate_config.yaml
#
# Via --export (no positional argument needed):
#   sbatch --export=ALL,CONFIG=/abs/path/to/animate_config.yaml submit_assemble.slurm
# =============================================================================
#SBATCH --job-name=anim_assemble
#SBATCH --output=logs_assemble_%j.out
#SBATCH --error=logs_assemble_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00

# ---------------------------------------------------------------------------
# Resolve CONFIG — priority order:
#   1. Positional argument $1 (sbatch submit_assemble.slurm /path/to/config.yaml)
#   2. $CONFIG environment variable  (--export=ALL,CONFIG=...)
#   3. Default: animate_config.yaml in the submission directory
# ---------------------------------------------------------------------------
CONFIG=${1:-${CONFIG:-"${SLURM_SUBMIT_DIR}/animate_config.yaml"}}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB
module load gcc/13.2.0
module load ffmpeg/6.0

python /fred/oz304/beckers/Chemistry/pipeline/plotting/animate_renders_v2.py assemble \
  --config "$CONFIG"
