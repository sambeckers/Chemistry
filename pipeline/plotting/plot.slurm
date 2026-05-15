#!/bin/bash
#SBATCH --job-name=dump_plot
#SBATCH --output=logs/dump_plot_%j.out
#SBATCH --error=logs/dump_plot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=work       # change to your cluster's partition name

# --- must match the array size used in slurm_accumulate.sh ---
N_TASKS=32
CHEMISTRY=${CHEMISTRY:-Crich}

echo "Plotting  chemistry=${CHEMISTRY}  n_tasks=${N_TASKS}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

python /fred/oz304/beckers/Chemistry/pipeline/plotting/plot_abmean_3dvs1d.py \
    --mode       plot         \
    --chemistry  "${CHEMISTRY}"     \
    --n-tasks    "${N_TASKS}"