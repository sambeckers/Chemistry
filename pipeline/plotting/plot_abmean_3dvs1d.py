#!/usr/bin/env python3
"""
plot_dump_vs_1d.py

Computes mean molecular abundances vs radius by accumulating ALL HDF5 dump
files from aphid, then plots those 3-D averages against the 1-D chemistry model.

SLURM workflow
--------------
Step 1 — accumulate (SLURM array, one task per chunk of dump files):

    sbatch slurm_accumulate.sh

    Each array task processes dump_files[task_id::n_tasks], accumulates
    per-bin sums and counts for every species, and writes a partial .npz
    to SCRATCH_DIR.

Step 2 — plot (single job, depends on step 1):

    sbatch --dependency=afterok:<jobid_step1> slurm_plot.sh

    Loads and sums all partial .npz files, applies log-space Gaussian
    smoothing, then produces two PDFs per chemistry:
        ab_mean_compare1D.pdf       — 2×2 panel (3-D top, 1-D bottom)
        ab_mean_compare1D_grid.pdf  — per-molecule grid (solid=3D, dashed=1D)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica",
})

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import BASE_PATH, daughters_Crich, daughters_Orich, parents
from code_chem.plotting.n_distinct_colours import generate_colormap
from kb.modeling.tools import CodeIO
from kb import path as kb_path
from code_chem.run_plot_single_config_1D_model import (
    run_model,
    find_star_for_config,
    get_fractional_abundance,
    TARGET_MLOSS,
    TARGET_VELOCITY,
)
from code_chem.plotting.plot_utils import apply_abundance_axis_limits, add_log_ticks


# ---------------------------------------------------------------------------
# Configuration — edit to match your setup
# ---------------------------------------------------------------------------

DUMP_DIR    = Path("/aphid/scratch-3month/sbeckers/v10a09_out/output/dumps")
SCRATCH_DIR = Path("/aphid/scratch-3month/sbeckers/v10a09_out/accum_scratch")  # partial .npz files
SAVE_DIR_BASE = BASE_PATH / "figures/v10a09_out"

# Radius domain (cm)
R_MIN = 1e13
R_MAX = 1e17
N_BINS = 300                  # log-spaced bins

# Gaussian smooth width in bins (log-spaced); set 0 to disable
SMOOTH_SIGMA = 0

# Bins with fewer particles than this are masked as NaN
MIN_COUNT = 30

# Threads per SLURM task
MAX_WORKERS = 32


# ---------------------------------------------------------------------------
# Pre-computed radius grid (constant; must match across all tasks)
# ---------------------------------------------------------------------------

R_EDGES   = np.logspace(np.log10(R_MIN), np.log10(R_MAX), N_BINS + 1)
R_CENTRES = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])   # geometric bin centre


# ---------------------------------------------------------------------------
# Species helpers
# ---------------------------------------------------------------------------

def get_all_species() -> list[str]:
    """Union of parents + both chemistry daughter lists."""
    return list(dict.fromkeys(parents + daughters_Crich + daughters_Orich))

def normalise_name(name: str) -> str:
    """Map a molecules.py name to its HDF5 dataset key (particles/<key>)."""
    cleaned = name.strip().lower()
    cleaned = cleaned.replace("+", "_plus")
    cleaned = cleaned.replace("-", "_minus")
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace("(", "_").replace(")", "_").replace(".", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned

# Built once at import; use everywhere you touch an HDF5 key
SPECIES_HDF5_KEY: dict[str, str] = {sp: normalise_name(sp) for sp in get_all_species()}

def get_species_and_colors(chemistry: str, present: set[str]):
    """Return (parent_list, daughter_list, color_dict) for species in *present*."""
    daughters = daughters_Orich if chemistry == "Orich" else daughters_Crich
    parent_species   = [s for s in parents   if s in present]
    daughter_species = [s for s in daughters if s in present]
    # de-duplicate: if a parent also appears in daughters, keep it only in parents
    all_species = parent_species + [s for s in daughter_species if s not in parent_species]
    cmap = generate_colormap(len(all_species)).colors
    colors = {sp: cmap[i] for i, sp in enumerate(all_species)}
    return parent_species, daughter_species, colors


# ---------------------------------------------------------------------------
# Per-dump worker — returns compact bin arrays, never the full particle array
# ---------------------------------------------------------------------------

def _accumulate_dump(path: Path, species_list: list[str]):
    """Read r + each species from one HDF5 dump; return per-bin sums/counts."""
    import h5py  # local import for thread-safety with some h5py builds

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        with h5py.File(path, "r") as f:
            r = f["particles/r"][:]
            for species in species_list:
                try:
                    ab = f[f"particles/{SPECIES_HDF5_KEY[species]}"][:]
                except KeyError:
                    continue
                mask = (
                    np.isfinite(r)  &
                    np.isfinite(ab) &
                    (r  > 0)        &
                    (ab > 0)        &
                    (ab <= 1.0)
                )
                n_total   = np.isfinite(ab).sum()
                n_unphysical = (ab > 1.0).sum()
                if n_unphysical > 0:
                    print(f"  {path.name} [{species}]: masked {n_unphysical}/{n_total} unphysical values")
                
                r_ok, ab_ok = r[mask], ab[mask]
                if r_ok.size == 0:
                    continue
                idx   = np.digitize(r_ok, R_EDGES) - 1
                valid = (idx >= 0) & (idx < N_BINS)
                result[species] = (
                    np.bincount(idx[valid], weights=ab_ok[valid], minlength=N_BINS),
                    np.bincount(idx[valid], minlength=N_BINS).astype(np.int64),
                )
    except Exception as exc:
        print(f"  Warning: skipping {path.name}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Chunk accumulation (one SLURM task)
# ---------------------------------------------------------------------------

def accumulate_chunk(
    dump_files: list[Path],
    species_list: list[str],
    max_workers: int = MAX_WORKERS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Accumulate bin sums/counts over *dump_files* using a thread pool."""
    totals: dict[str, list] = {
        s: [np.zeros(N_BINS), np.zeros(N_BINS, dtype=np.int64)]
        for s in species_list
    }
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_accumulate_dump, p, species_list): p for p in dump_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Dumps"):
            for sp, (s, c) in fut.result().items():
                totals[sp][0] += s
                totals[sp][1] += c
    return {sp: (totals[sp][0], totals[sp][1]) for sp in species_list}


# ---------------------------------------------------------------------------
# SLURM mode: accumulate
# ---------------------------------------------------------------------------

def run_accumulate(task_id: int, n_tasks: int, chemistry: str) -> None:
    """Process this task's slice of dump files and write a partial .npz."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    dump_files = sorted(DUMP_DIR.glob("dump_*.h5"))
    if not dump_files:
        raise FileNotFoundError(f"No dump_*.h5 files found in {DUMP_DIR}")

    # Strided split so each task gets a balanced sample across the time series
    chunk = dump_files[task_id::n_tasks]
    print(f"Task {task_id + 1}/{n_tasks}: processing {len(chunk)}/{len(dump_files)} dumps")

    species_list = get_all_species()
    totals = accumulate_chunk(chunk, species_list)

    out_path = SCRATCH_DIR / f"partial_{chemistry}_{task_id:04d}_of_{n_tasks:04d}.npz"
    save_dict: dict[str, np.ndarray] = {}
    for sp, (s, c) in totals.items():
        # Encode species name in key; "__" is the separator
        save_dict[f"{sp}__sum"]   = s
        save_dict[f"{sp}__count"] = c
    np.savez(out_path, **save_dict)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Aggregation: merge all partial .npz files
# ---------------------------------------------------------------------------

def aggregate_partials(
    chemistry: str,
    n_tasks: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pattern = f"partial_{chemistry}_*_of_{n_tasks:04d}.npz"
    files   = sorted(SCRATCH_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No partial files matching '{pattern}' in {SCRATCH_DIR}\n"
            "Run the accumulate step first."
        )
    print(f"Aggregating {len(files)} partial files …")

    totals: dict[str, list] = {}
    for path in files:
        data = np.load(path)
        for key in data.files:
            sp, kind = key.rsplit("__", 1)
            if sp not in totals:
                totals[sp] = [np.zeros(N_BINS), np.zeros(N_BINS, dtype=np.int64)]
            if kind == "sum":
                totals[sp][0] += data[key]
            elif kind == "count":
                totals[sp][1] += data[key]
    return {sp: (totals[sp][0], totals[sp][1]) for sp in totals}


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_mean(
    bin_sum: np.ndarray,
    bin_count: np.ndarray,
    min_count: int = MIN_COUNT,
    sigma: float = SMOOTH_SIGMA,
) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(bin_count >= min_count, bin_sum / bin_count, np.nan)
    if sigma <= 0:
        return mean
    log_mean = np.log10(mean)
    nan_mask = ~np.isfinite(log_mean)
    if nan_mask.all():
        return mean
    x = np.arange(N_BINS)
    log_filled   = np.interp(x, x[~nan_mask], log_mean[~nan_mask])
    log_smoothed = gaussian_filter1d(log_filled, sigma=sigma)
    log_smoothed[nan_mask] = np.nan
    result = 10.0 ** log_smoothed
    result = np.where(result > 1.0, np.nan, result)  # enforce physical ceiling
    return result


# ---------------------------------------------------------------------------
# 1-D model loader (unchanged from original)
# ---------------------------------------------------------------------------

def load_1d_data(chemistry: str):
    if chemistry == "Crich":
        inputfile = str(BASE_PATH / "KoekenBak/input/20251015_Sam_Mdot_Vinf_Crich.dat")
    else:
        inputfile = str(BASE_PATH / "KoekenBak/input/20251015_Sam_Mdot_Vinf_Orich.dat")
    model = run_model(inputfile)
    idx, star, mloss_label, vinf_label = find_star_for_config(model, TARGET_MLOSS, TARGET_VELOCITY)
    folder = os.path.join(kb_path.cout, "models", star["LAST_CHEMISTRY_MODEL"]) + "/"
    radius_1d = CodeIO.getChemistryPhysPar(folder + "csphyspar_smooth.out", "RADIUS")
    fracs_1d  = CodeIO.getChemistryAbundances(folder + "csfrac_smooth.out")
    return radius_1d, fracs_1d, mloss_label, vinf_label


# ---------------------------------------------------------------------------
# Plot 1 — 2×2 panel: 3-D top row, 1-D bottom row
# ---------------------------------------------------------------------------

def plot_avg_abundances_compare_1d(
    r_grid, averages, contributors, chemistry, save_path, show=False
):
    parent_species, daughter_species, species_colors = get_species_and_colors(
        chemistry, set(averages)
    )
    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    ax_par_3d, ax_dau_3d = axes[0, 0], axes[0, 1]
    ax_par_1d, ax_dau_1d = axes[1, 0], axes[1, 1]

    for sp in parent_species:
        ax_par_3d.plot(r_grid, averages[sp], lw=3, color=species_colors[sp],
                       label=f"{sp} (N={contributors[sp]:,})")
    for sp in daughter_species:
        ax_dau_3d.plot(r_grid, averages[sp], lw=3, color=species_colors[sp],
                       label=f"{sp} (N={contributors[sp]:,})")

    ax_par_3d.set_title("Parents — 3D average", fontsize=14)
    ax_dau_3d.set_title("Daughters — 3D average", fontsize=14)

    for sp in parent_species:
        frac = get_fractional_abundance(fracs_1d, sp)
        if frac is None:
            print(f"1-D: missing parent {sp}")
            continue
        ax_par_1d.plot(radius_1d, frac, lw=3, color=species_colors[sp], label=sp)
    for sp in daughter_species:
        frac = get_fractional_abundance(fracs_1d, sp)
        if frac is None:
            print(f"1-D: missing daughter {sp}")
            continue
        ax_dau_1d.plot(radius_1d, frac, lw=3, color=species_colors[sp], label=sp)

    ax_par_1d.set_title(f"Parents — 1D ({mloss_label}, {vinf_label})", fontsize=14)
    ax_dau_1d.set_title(f"Daughters — 1D ({mloss_label}, {vinf_label})", fontsize=14)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9, ncol=3)
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)
        ax.set_xlabel("Radius [cm]", fontsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel("Abundance (wrt H$_{nuc}$)", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — per-molecule grid: solid = 3-D, dashed = 1-D
# ---------------------------------------------------------------------------

def plot_compare_1d_grid(
    r_grid, averages, contributors, chemistry, save_path,
    show=False, n_per_panel=3,
):
    parent_species, daughter_species, species_colors = get_species_and_colors(
        chemistry, set(averages)
    )
    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)

    all_species = parent_species + [s for s in daughter_species if s not in parent_species]
    groups  = [all_species[i:i + n_per_panel] for i in range(0, len(all_species), n_per_panel)]
    n_panels = len(groups)
    n_cols   = min(4, n_panels)
    n_rows   = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.2 * n_cols, 4.5 * n_rows),
        dpi=300, sharex=True, sharey=False,
    )
    axes_arr  = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes_arr.flatten()

    for panel_idx, group in enumerate(groups):
        ax = axes_flat[panel_idx]
        for sp in group:
            color = species_colors[sp]
            if sp in averages:
                ax.plot(r_grid, averages[sp], lw=3, color=color, ls="-",
                        label=f"{sp} (N={contributors.get(sp, 0):,})")
            frac_1d = get_fractional_abundance(fracs_1d, sp)
            if frac_1d is not None:
                ax.plot(radius_1d, frac_1d, lw=3, color=color, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9, ncol=1)
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # x-label: bottom-most visible panel per column
    for col_idx in range(n_cols):
        for row_idx in range(n_rows - 1, -1, -1):
            if row_idx * n_cols + col_idx < n_panels:
                axes_arr[row_idx, col_idx].set_xlabel("Radius [cm]", fontsize=13)
                break

    # y-label: left column only
    for row_idx in range(n_rows):
        if row_idx * n_cols < n_panels:
            axes_arr[row_idx, 0].set_ylabel(
                r"Abundance (wrt H$_{\mathrm{nuc}}$)", fontsize=13
            )

    style_handles = [
        Line2D([0], [0], color="k", lw=3, ls="-",  label="3D average"),
        Line2D([0], [0], color="k", lw=3, ls="--", label=f"1D ({mloss_label}, {vinf_label})"),
    ]
    fig.legend(handles=style_handles, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 0.0), frameon=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# SLURM mode: plot
# ---------------------------------------------------------------------------

def run_plot(chemistry: str, n_tasks: int, show: bool = False) -> None:
    """Aggregate partials → smooth → produce comparison plots."""
    totals = aggregate_partials(chemistry, n_tasks)

    averages: dict[str, np.ndarray]  = {}
    contributors: dict[str, int]     = {}
    for sp, (bin_sum, bin_count) in totals.items():
        averages[sp]     = smooth_mean(bin_sum, bin_count)
        contributors[sp] = int(bin_count.sum())
        n_pop = int((bin_count >= MIN_COUNT).sum())
        print(f"  {sp}: {n_pop}/{N_BINS} bins above MIN_COUNT={MIN_COUNT}, "
              f"total particles = {contributors[sp]:,}")

    save_dir = SAVE_DIR_BASE / chemistry
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_avg_abundances_compare_1d(
        R_CENTRES, averages, contributors, chemistry,
        save_dir / "ab_mean_compare1D.png",
        show=show,
    )
    print(f"Saved: {save_dir / 'ab_mean_compare1D.png'}")

    plot_compare_1d_grid(
        R_CENTRES, averages, contributors, chemistry,
        save_dir / "ab_mean_compare1D_grid.png",
        show=show,
    )
    print(f"Saved: {save_dir / 'ab_mean_compare1D_grid.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False
    
def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D abundance averaging vs 1-D chemistry model."
    )
    parser.add_argument(
        "--mode", choices=["accumulate", "plot"], required=True,
        help=(
            "'accumulate': process a chunk of dump files and write a partial .npz  "
            "(run as SLURM array job).  "
            "'plot': aggregate all partials and produce comparison PDFs  "
            "(run after the array completes)."
        ),
    )
    parser.add_argument(
        "--chemistry", choices=["Crich", "Orich"], default="Crich",
    )
    parser.add_argument(
        "--task-id", type=int, default=0,
        help="0-indexed SLURM_ARRAY_TASK_ID (accumulate mode only).",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=1,
        help="Total number of array tasks; must match between accumulate and plot steps.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively (plot mode only; requires a display).",
    )

    if is_interactive():
        # ── Edit these defaults for interactive/notebook use ──
        args = parser.parse_args([
            "--mode",      "plot",
            "--chemistry", "Crich",
            "--n-tasks",   "32",
            "--show",
        ])
    else:
        args = parser.parse_args()

    if args.mode == "accumulate":
        run_accumulate(args.task_id, args.n_tasks, args.chemistry)
    else:
        run_plot(args.chemistry, args.n_tasks, show=args.show)


if __name__ == "__main__":
    main()
