#!/bin/bash
#SBATCH --job-name=mol_assemble
#SBATCH --output=logs_%A/molecule_art_%A_%a.out
#SBATCH --error=logs_%A/molecule_art_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=16G

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

python /fred/oz304/beckers/Chemistry/pipeline/plotting/molecule_grid_art.py assemble \
    --panel-dir   /fred/oz304/beckers/Chemistry/figures/v10a09_out/molecule_art/panels \
    --output-stem /fred/oz304/beckers/Chemistry/figures/v10a09_out/molecule_art/molecule_art \
    --dump-dir     "/fred/oz304/beckers/v10a09_out/output/dumps" \
    --phantom-dir  "/fred/oz304/beckers/v10a09" \
    --orientation both