#!/bin/bash
# =============================================================================
# submit_molecule_art.sh
#
# Single command to render all molecular column‑density panels in parallel
# (array job) and then automatically assemble them into A4 artwork grids.
#
# Usage:
#   bash submit_molecule_art.sh
#
# You can override the default parameters before running, e.g.:
#   DUMP_INDEX=1581 PLANE=xz bash submit_molecule_art.sh
# =============================================================================

# ----- Default parameters (change as needed) ---------------------------------
DUMP_INDEX="${DUMP_INDEX:-1581}"
CHEMISTRY="${CHEMISTRY:-Crich}"
PLANE="${PLANE:-xy}"
XLIM="${XLIM:-100}"
CMAP="${CMAP:-gist_heat}"
N_TASKS="${N_TASKS:-20}"          # number of array tasks; increase if needed

# ----- Paths -----------------------------------------------------------------
SCRIPT_DIR="/fred/oz304/beckers/Chemistry/pipeline/plotting"
PANEL_DIR="/fred/oz304/beckers/Chemistry/figures/v10a09_out/molecule_art/panels"
FINAL_STEM="/fred/oz304/beckers/Chemistry/figures/v10a09_out/molecule_art/molecule_art"
LOG_DIR="logs_molecule_art_${DUMP_INDEX}"

mkdir -p "${LOG_DIR}"

# =============================================================================
# 1. Submit the array job (panels rendering)
# =============================================================================
echo "Submitting array job for panel rendering..."
ARRAY_JOBID=$(sbatch --parsable <<-EOF
#!/bin/bash
#SBATCH --job-name=mol_art_panels
#SBATCH --output=${LOG_DIR}/array_%A_%a.out
#SBATCH --error=${LOG_DIR}/array_%A_%a.err
#SBATCH --array=0-$((N_TASKS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

python "${SCRIPT_DIR}/molecule_grid_art.py" panels \
    --dump-index   ${DUMP_INDEX} \
    --dump-dir     "/fred/oz304/beckers/v10a09_out/output/dumps" \
    --phantom-dir  "/fred/oz304/beckers/v10a09" \
    --chemistry    ${CHEMISTRY} \
    --plane        ${PLANE} \
    --xlim         ${XLIM} \
    --cmap         ${CMAP} \
    --output-dir   "${PANEL_DIR}" \
    --task-id      "\${SLURM_ARRAY_TASK_ID}" \
    --n-tasks      ${N_TASKS}
EOF
)

if [ -z "$ARRAY_JOBID" ]; then
    echo "ERROR: Failed to submit array job."
    exit 1
fi
echo "Array job submitted with ID: $ARRAY_JOBID"

# =============================================================================
# 2. Submit the assembly job, dependent on successful completion of the array
# =============================================================================
echo "Submitting assembly job (dependency: afterany:$ARRAY_JOBID)..."
ASSEMBLY_JOBID=$(sbatch --parsable --dependency=afterany:$ARRAY_JOBID <<-EOF
#!/bin/bash
#SBATCH --job-name=mol_art_assemble
#SBATCH --output=${LOG_DIR}/assemble_%j.out
#SBATCH --error=${LOG_DIR}/assemble_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fred/oz304/beckers/KB

python "${SCRIPT_DIR}/molecule_grid_art.py" assemble \
    --panel-dir   "${PANEL_DIR}" \
    --output-stem "${FINAL_STEM}" \
    --orientation both
EOF
)

if [ -z "$ASSEMBLY_JOBID" ]; then
    echo "ERROR: Failed to submit assembly job."
    exit 1
fi
echo "Assembly job submitted with ID: $ASSEMBLY_JOBID"
echo "Done. The assembly will run automatically after all panel tasks finish."