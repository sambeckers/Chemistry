#!/bin/bash
# =============================================================================
# submit_assemble.slurm
#
# Submit the assembly step as a short job.
# Pass the YAML config file as a command‑line argument.
#
# Usage: sbatch submit_assemble.slurm animate_config.yaml
# =============================================================================
#SBATCH --job-name=anim_assemble
#SBATCH --output=logs_assemble_%j.out
#SBATCH --error=logs_assemble_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00

CONFIG=${1:-animate_config.yaml}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

python /fred/oz304/beckers/Chemistry/pipeline/plotting/animate_renders_v2.py assemble --config "$CONFIG"