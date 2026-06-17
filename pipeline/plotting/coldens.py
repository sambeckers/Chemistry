#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import os
import contextlib
import io

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_PATH
from code_chem.plotting.plot_utils import apply_abundance_axis_limits, add_log_ticks, set_plot_style, _collect_xy, format_species_label

from code_chem.run_plot_single_config_1D_model import (
    run_model,
    find_star_for_config,
    get_fractional_abundance,
    TARGET_MLOSS,
    TARGET_VELOCITY,
)
from kb.modeling.tools import CodeIO
from kb import path as kb_path

DUMP_DIR      = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
SCRATCH_DIR   = Path("/aphid/scratch-3month/sbeckers/v10a09_out/accum_scratch")
SAVE_DIR_BASE = BASE_PATH / "figures/v10a09_out"

R_MIN, R_MAX, N_BINS = 1e14, 1e17, 300
R_EDGES   = np.logspace(np.log10(R_MIN), np.log10(R_MAX), N_BINS + 1)
R_CENTRES = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])

# Abundance axis (species)
N_AB_BINS  = 150
AB_LOG_MIN = -20.0
AB_LOG_MAX =  0.0
AB_EDGES   = np.logspace(AB_LOG_MIN, AB_LOG_MAX, N_AB_BINS + 1)
AB_CENTRES = np.sqrt(AB_EDGES[:-1] * AB_EDGES[1:])

MIN_COUNT    = 30
SMOOTH_SIGMA = 0       # set > 0 to smooth the overlay lines

PHYS_PARAMS: dict[str, dict] = {
    "temperature": {
        "hdf5_key":  "temp",           # particles/<hdf5_key>
        "log_min":  0.0,    # 1 K
        "log_max":  6.0,    # 1 000 000 K
        "n_bins":   150,
        "label":    "Temperature [K]",
        "color":    "#d6604d",
    },
    "av": {
        "hdf5_key":  "av",
        "log_min": -6.0,    # 10^-6 mag
        "log_max":  1.0,    # 10 mag
        "n_bins":   150,
        "label":    r"$A_V$ [mag]",
        "color":    "#4dac26",
    },
    "density": {
        "hdf5_key":  "density",
        "log_min": -1.0,    # 0.1 cm^-3
        "log_max": 10.0,    # 10^10 cm^-3
        "n_bins":  150,
        "label":   r"$n$ [cm$^{-3}$]",
        "color":   "#762a83",
    },
}

for _cfg in PHYS_PARAMS.values():
    _cfg["edges"]   = np.logspace(_cfg["log_min"], _cfg["log_max"],
                                  _cfg["n_bins"] + 1)
    _cfg["centres"] = np.sqrt(_cfg["edges"][:-1] * _cfg["edges"][1:])


def load_1d_data(chemistry: str):
    if chemistry == "Crich":
        inputfile = str(BASE_PATH /
                        "KoekenBak/input/20251015_Sam_Mdot_Vinf_Crich.dat")
    else:
        inputfile = str(BASE_PATH /
                        "KoekenBak/input/20251015_Sam_Mdot_Vinf_Orich.dat")
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
            model = run_model(inputfile)
            idx, star, mloss_label, vinf_label = find_star_for_config(
                model, TARGET_MLOSS, TARGET_VELOCITY
            )
            folder = (os.path.join(kb_path.cout, "models",
                                star["LAST_CHEMISTRY_MODEL"]) + "/")
            radius_1d = CodeIO.getChemistryPhysPar(
                folder + "csphyspar_smooth.out", "RADIUS"
            )
            fracs_1d  = CodeIO.getChemistryAbundances(folder + "csfrac_smooth.out")
            coldens = CodeIO.getColumnDensities(folder + "cscoldens_smooth.out")
            nH2 = CodeIO.getChemistryPhysPar(folder + "csphyspar_smooth.out", "n(H2)")

    print(f"Loaded 1D model")
    return radius_1d, fracs_1d, coldens, nH2, mloss_label, vinf_label

def aggregate_molecule(
    molecule: str,
    chemistry: str,
    n_tasks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate partial .npz files for *molecule*.

    Returns
    -------
    bin_sum   : (N_BINS,)
    bin_count : (N_BINS,)
    hist_2d   : (N_BINS, N_AB_BINS)
    """
    pattern = f"partial_{chemistry}_*_of_{n_tasks:04d}.npz"
    files   = sorted(SCRATCH_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No partial files matching '{pattern}' in {SCRATCH_DIR}\n"
            "Run the accumulate step of plot_stats.py first."
        )
    print(f"Aggregating {len(files)} partial files for '{molecule}' …")

    bin_sum   = np.zeros(N_BINS, dtype=np.float64)
    bin_count = np.zeros(N_BINS, dtype=np.int64)
    hist_2d   = np.zeros((N_BINS, N_AB_BINS), dtype=np.int64)

    found = False
    for path in files:
        data = np.load(path)
        sum_key   = f"{molecule}__sum"
        count_key = f"{molecule}__count"
        hist_key  = f"{molecule}__hist"
        if sum_key not in data.files:
            continue
        found = True
        bin_sum   += data[sum_key]
        bin_count += data[count_key]
        hist_2d   += data[hist_key]

    if not found:
        sample    = np.load(files[0])
        available = sorted({k.rsplit("__", 1)[0] for k in sample.files
                            if not k.startswith("phys__")})
        raise KeyError(
            f"Molecule '{molecule}' not found in partial files.\n"
            f"Available species: {available}"
        )

    return bin_sum, bin_count, hist_2d


# ---------------------------------------------------------------------------
# Load & aggregate — physical parameters
# ---------------------------------------------------------------------------

def aggregate_phys_param(
    param: str,
    chemistry: str,
    n_tasks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate partial .npz files for a physical parameter.

    Keys in the .npz follow the 'phys__<param>__sum/count/hist' convention
    written by plot_stats.py's run_accumulate().

    Returns
    -------
    bin_sum   : (N_BINS,)
    bin_count : (N_BINS,)
    hist_2d   : (N_BINS, cfg["n_bins"])
    """
    if param not in PHYS_PARAMS:
        raise ValueError(
            f"Unknown physical parameter '{param}'. "
            f"Choose from: {list(PHYS_PARAMS)}"
        )
    cfg     = PHYS_PARAMS[param]
    pattern = f"partial_{chemistry}_*_of_{n_tasks:04d}.npz"
    files   = sorted(SCRATCH_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No partial files matching '{pattern}' in {SCRATCH_DIR}\n"
            "Run the accumulate step of plot_stats.py first."
        )
    print(f"Aggregating {len(files)} partial files for phys param '{param}' …")

    bin_sum   = np.zeros(N_BINS, dtype=np.float64)
    bin_count = np.zeros(N_BINS, dtype=np.int64)
    hist_2d   = np.zeros((N_BINS, cfg["n_bins"]), dtype=np.int64)

    found = False
    for path in files:
        data      = np.load(path)
        sum_key   = f"phys__{param}__sum"
        count_key = f"phys__{param}__count"
        hist_key  = f"phys__{param}__hist"
        if sum_key not in data.files:
            continue
        found = True
        bin_sum   += data[sum_key]
        bin_count += data[count_key]
        hist_2d   += data[hist_key]

    if not found:
        raise KeyError(
            f"Physical parameter '{param}' not found in partial files. "
            "Were the .npz files produced with a version of plot_stats.py "
            "that includes physical parameter accumulation?"
        )

    return bin_sum, bin_count, hist_2d

def _smooth(
    arr: np.ndarray,
    sigma: float = SMOOTH_SIGMA,
    clip_at_one: bool = False,
) -> np.ndarray:
    """
    Log-space Gaussian smoothing.

    Parameters
    ----------
    clip_at_one : bool
        Set True for fractional abundances (must be ≤ 1).
        Leave False for physical parameters with no upper bound.
    """
    if sigma <= 0:
        return arr
    log_arr  = np.log10(arr)
    nan_mask = ~np.isfinite(log_arr)
    if nan_mask.all():
        return arr
    x        = np.arange(len(arr))
    filled   = np.interp(x, x[~nan_mask], log_arr[~nan_mask])
    smoothed = gaussian_filter1d(filled, sigma=sigma)
    smoothed[nan_mask] = np.nan
    out = 10.0 ** smoothed
    if clip_at_one:
        out = np.where(out > 1.0, np.nan, out)
    return out

def compute_stats(
    bin_sum:     np.ndarray,
    bin_count:   np.ndarray,
    hist_2d:     np.ndarray,
    val_centres: np.ndarray,
    clip_at_one: bool = False,
) -> dict[str, np.ndarray]:
    """
    Return mean, median, p16, p84 — each shape (N_BINS,).

    Parameters
    ----------
    val_centres : (n_val_bins,)
        Geometric centres of the value axis.  For species use AB_CENTRES;
        for physical parameters use cfg["centres"].
    clip_at_one : bool
        Passed through to _smooth(); True for abundances, False for
        physical parameters.
    """
    n_val_bins = hist_2d.shape[1]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(bin_count >= MIN_COUNT,
                        bin_sum / bin_count, np.nan)

    row_counts = hist_2d.sum(axis=1).astype(np.float64)
    cdf = np.cumsum(hist_2d.astype(np.float64), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(
            row_counts[:, None] >= MIN_COUNT,
            cdf / row_counts[:, None],
            np.nan,
        )

    def _percentile(frac: float) -> np.ndarray:
        out = np.full(N_BINS, np.nan)
        for i in range(N_BINS):
            if row_counts[i] < MIN_COUNT:
                continue
            idx = np.searchsorted(norm[i], frac, side="left")
            if idx < n_val_bins:
                out[i] = val_centres[idx]
        return out

    return {
        "mean": _smooth(mean,                clip_at_one=clip_at_one),
        "p50":  _smooth(_percentile(0.50),   clip_at_one=clip_at_one),
        "p16":  _smooth(_percentile(0.16),   clip_at_one=clip_at_one),
        "p84":  _smooth(_percentile(0.84),   clip_at_one=clip_at_one),
    }

def calc_int_1D_coldens(molecule: str, radius_1d: np.ndarray, fracs_1d: dict[str, np.ndarray], nH2: np.ndarray) -> float:
    frac = get_fractional_abundance(fracs_1d, molecule)
    nMol = nH2 * frac
    return np.trapz(nMol, radius_1d)


def interp_1d_nH2_onto_grid(radius_1d: np.ndarray, nH2_1d: np.ndarray) -> np.ndarray:
    """
    Interpolate the 1D nH2 profile onto R_CENTRES using log-log interpolation.

    Grid points outside the 1D radial range are set to NaN.

    Parameters
    ----------
    radius_1d : (M,)  radial grid of the 1D model [cm]
    nH2_1d    : (M,)  H2 number density on that grid [cm^-3]

    Returns
    -------
    nH2_grid  : (N_BINS,)  nH2 interpolated onto R_CENTRES
    """
    log_r_1d   = np.log10(radius_1d)
    log_nH2_1d = np.log10(nH2_1d)
    log_r_grid = np.log10(R_CENTRES)
    log_nH2_interp = np.interp(
        log_r_grid, log_r_1d, log_nH2_1d,
        left=np.nan, right=np.nan,
    )
    return 10.0 ** log_nH2_interp


def calc_3D_int_coldens(
    molecule: str,
    chemistry: str,
    n_tasks: int,
    nH2_override: np.ndarray | None = None,
) -> float:
    """
    Integrate the 3D median fractional abundance against a density profile.

    Parameters
    ----------
    nH2_override : (N_BINS,) or None
        If provided, use this nH2 profile (already on R_CENTRES) instead of
        the 3D median density.  Pass the output of
        ``interp_1d_nH2_onto_grid`` to mix 3D chemistry with the 1D density.
    """
    bin_sum, bin_count, hist_2d = aggregate_molecule(
        molecule, chemistry, n_tasks)

    median_frac = compute_stats(
        bin_sum, bin_count, hist_2d,
        val_centres=AB_CENTRES,
        clip_at_one=True)["p50"]

    if nH2_override is not None:
        nH2 = nH2_override
    else:
        param = "density"
        bin_sum, bin_count, hist_2d = aggregate_phys_param(
            param, chemistry, n_tasks)
        nH2 = compute_stats(
            bin_sum, bin_count, hist_2d,
            val_centres=PHYS_PARAMS[param]["centres"],
            clip_at_one=False)["p50"]

    mask = np.isfinite(median_frac) & np.isfinite(nH2)

    return np.trapz(nH2[mask] * median_frac[mask], R_CENTRES[mask])


def plot_column_densities(
    molecules: list[str],
    cd_1d: dict[str, float],
    cd_3d: dict[str, float],
    chemistry: str,
    save_path: Path | None = None,
    cd_3d_1d_nh2: dict[str, float] | None = None,
) -> plt.Figure:
    set_plot_style()
    cd_active = cd_3d_1d_nh2 if cd_3d_1d_nh2 is not None else cd_3d

    # Sort molecules by log10(3D / 1D) descending (3D > 1D first).
    def _ratio(m: str) -> float:
        v1 = cd_1d.get(m, np.nan)
        v3 = cd_active.get(m, np.nan)
        if np.isfinite(v1) and np.isfinite(v3) and v1 > 0 and v3 > 0:
            return np.log10(v3 / v1)
        return -np.inf

    molecules = sorted(molecules, key=_ratio, reverse=True)

    x    = np.arange(len(molecules))
    y_1d = np.array([cd_1d.get(m, np.nan) for m in molecules])

    if cd_3d_1d_nh2 is not None:
        y_3d      = np.array([cd_3d_1d_nh2.get(m, np.nan) for m in molecules])
        label_3d  = r"3D (w/ 1D $n^{\mathrm{H_2}}$ profile)"
        marker_3d = "D"
        color_3d  = "#2ca02c"
    else:
        y_3d      = np.array([cd_3d.get(m, np.nan) for m in molecules])
        label_3d  = "3D (median)"
        marker_3d = "s"
        color_3d  = "tomato"

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.set_yscale("log")

    # Dashed vertical line at the 3D>1D / 3D<1D transition
    ratios      = np.array([_ratio(m) for m in molecules])
    cross_idx   = np.searchsorted(-ratios, 0)   # first index where ratio < 0
    if 0 < cross_idx < len(molecules):
        ax.axvline(cross_idx - 0.5, color="grey", lw=1.0,
                   ls="--", zorder=2, alpha=0.7)

    # ------------------------------------------------------------------ #
    # Connector lines coloured by direction                               #
    # ------------------------------------------------------------------ #
    for xi, v1, v3 in zip(x, y_1d, y_3d):
        if np.isfinite(v1) and np.isfinite(v3):
            ax.plot([xi, xi], [v1, v3],
                    color='k', lw=1.2, zorder=3, alpha=0.85)

    # ------------------------------------------------------------------ #
    # Scatter points (drawn on top)                                       #
    # ------------------------------------------------------------------ #
    ax.scatter(x, y_1d,
               marker="o", s=70, zorder=5,
               color="steelblue", label="1D model")
    ax.scatter(x, y_3d,
               marker=marker_3d, s=60, zorder=5,
               color=color_3d, label=label_3d)

    # ------------------------------------------------------------------ #
    # Axes / labels                                                        #
    # ------------------------------------------------------------------ #
    ax.set_xticks(x)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xticklabels(
        [format_species_label(m) for m in molecules],
        rotation=45, ha="right", rotation_mode="anchor", fontsize=16,
    )
    ax.set_ylabel(r"Column density [cm$^{-2}$]", fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlim(-0.5, len(molecules) - 0.5)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved column-density comparison to {save_path}")

    return fig


def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--molecule",
        nargs="+",
        type=str,
        help="Chemical species to plot (e.g. CO HCl SiO).",
    )
    parser.add_argument("--chemistry", choices=["Crich", "Orich"],
                        default="Crich")
    parser.add_argument(
        "--n-tasks", type=int, default=1,
        help="Must match the n-tasks used in the accumulate step.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the column-density comparison figure to SAVE_DIR_BASE.",
    )
    parser.add_argument(
        "--use-1d-nh2", action="store_true",
        help=(
            "Also compute a hybrid column density using the 3D median "
            "fractional abundance combined with the 1D nH2 profile "
            "(interpolated onto the 3D radial grid).  Adds a third series "
            "to the plot."
        ),
    )
    parents = ["CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN",
           "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS"]
    daughters = ['CN', 'CH2', 'CH3', 'C2H', 'C4H', 'C6H', 'CH3CN', 
                 'HC3N', 'HC5N', 'HC7N', 'H2CS', 'H2CO',  "SiC", "SiN"]
    if is_interactive():
        args = parser.parse_args([
            "--molecule",       *daughters,
            "--chemistry",      "Crich",
            "--use-1d-nh2",
            "--n-tasks",        "32",
            "--save",
        ])
    else:
        args = parser.parse_args()

    radius_1d, fracs_1d, coldens, nH2, mloss_label, vinf_label = load_1d_data("Crich")

    # Interpolate 1D nH2 onto the 3D radial grid once, reused for every molecule.
    nH2_on_grid = interp_1d_nH2_onto_grid(radius_1d, nH2) if args.use_1d_nh2 else None

    if args.molecule:
        cd_1d: dict[str, float] = {}
        cd_3d: dict[str, float] = {}
        cd_3d_1d_nh2: dict[str, float] = {} if args.use_1d_nh2 else None

        for molecule in args.molecule:
            cd_1d[molecule] = coldens[molecule]
            cd_3d[molecule] = calc_3D_int_coldens(
                molecule, args.chemistry, args.n_tasks
            )
            print(f"{molecule}: 1D column density        = {cd_1d[molecule]:.3e} cm^-3")
            print(f"{molecule}: 3D column density        = {cd_3d[molecule]:.3e} cm^-3")

            if args.use_1d_nh2:
                cd_3d_1d_nh2[molecule] = calc_3D_int_coldens(
                    molecule, args.chemistry, args.n_tasks,
                    nH2_override=nH2_on_grid,
                )
                print(f"{molecule}: 3D frac × 1D nH2         = {cd_3d_1d_nh2[molecule]:.3e} cm^-3")

        save_dir = SAVE_DIR_BASE / args.chemistry
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_1d_nh2" if args.use_1d_nh2 else ""
        if args.save:
            save_path = (
                save_dir / f"coldens_comparison_daughters{suffix}.png"
            )

        fig = plot_column_densities(
            molecules=args.molecule,
            cd_1d=cd_1d,
            cd_3d=cd_3d,
            chemistry=args.chemistry,
            save_path=save_path,
            cd_3d_1d_nh2=cd_3d_1d_nh2,
        )
        plt.show()


if __name__ == "__main__":
    main()