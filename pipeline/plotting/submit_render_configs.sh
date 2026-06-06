#!/bin/bash
# =============================================================================
# submit_render_configs.sh
#
# Expands configs_render.csv into a flat task list (one line per
# plane × xlim × xsec combination) and submits render_configs.slurm
# as an array job — one task per combination.
#
# Usage:
#   bash submit_render_configs.sh <dump_index> [chemistry] [cmap]
#
# Examples:
#   bash submit_render_configs.sh 1581
#   bash submit_render_configs.sh 1581 Orich
#   bash submit_render_configs.sh 1581 Crich gist_heat
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_CSV="${SCRIPT_DIR}/configs_render.csv"
SLURM_SCRIPT="${SCRIPT_DIR}/render_configs.slurm"

# --- Arguments ---------------------------------------------------------------
DUMP_INDEX="${1:-}"
CHEMISTRY="${2:-Crich}"
CMAP="${3:-gist_heat}"

if [ -z "${DUMP_INDEX}" ]; then
    echo "Usage: bash submit_render_configs.sh <dump_index> [chemistry] [cmap]"
    exit 1
fi

# --- Generate flat task list -------------------------------------------------
# Each line: row_idx,plane,xlim,xsec
# Uses a minimal Python script — no heavy imports, fast even on login nodes.

TASK_LIST="${SCRIPT_DIR}/task_list_${DUMP_INDEX}.txt"

python3 - "${CONFIG_CSV}" "${TASK_LIST}" <<'PYEOF'
import csv, sys
from itertools import product

config_csv = sys.argv[1]
task_list  = sys.argv[2]

with open(config_csv, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

tasks = []
for row_idx, row in enumerate(rows):
    # planes
    planes = [p.strip() for p in row.get("plane", "xy").split(",") if p.strip()]

    # xlims
    xlim_raw = row.get("xlim", "").strip()
    xlims = ["None"] if xlim_raw in ("", "None") \
            else [v.strip() for v in xlim_raw.split(",")]

    # xsecs
    xsec_raw = row.get("xsec", "").strip()
    xsecs = ["None"] if xsec_raw in ("", "None") \
            else [v.strip() for v in xsec_raw.split(",")]

    for plane, xlim, xsec in product(planes, xlims, xsecs):
        tasks.append(f"{row_idx},{plane},{xlim},{xsec}")

with open(task_list, "w") as f:
    f.write("\n".join(tasks) + "\n")

print(f"Generated {len(tasks)} tasks -> {task_list}", flush=True)
for i, t in enumerate(tasks):
    print(f"  [{i:3d}] {t}", flush=True)
PYEOF

# --- Count tasks -------------------------------------------------------------
N_TASKS=$(grep -c . "${TASK_LIST}")
LAST_IDX=$(( N_TASKS - 1 ))

echo
echo "Config CSV  : ${CONFIG_CSV}"
echo "Task list   : ${TASK_LIST}"
echo "Tasks total : ${N_TASKS}  (array 0-${LAST_IDX})"
echo "Dump index  : ${DUMP_INDEX}"
echo "Chemistry   : ${CHEMISTRY}"
echo "Cmap        : ${CMAP}"
echo

# --- Submit ------------------------------------------------------------------
sbatch \
    --array="0-${LAST_IDX}" \
    --export="ALL,DUMP_INDEX=${DUMP_INDEX},CHEMISTRY=${CHEMISTRY},CMAP=${CMAP},TASK_LIST=${TASK_LIST}" \
    "${SLURM_SCRIPT}"
