#!/usr/bin/env python3
"""
plot_pointdensity.py

Visualises the 2-D point density (radius × value) for a single molecule
OR a physical parameter (temperature, AV, density) by reading the partial
.npz files produced by the accumulate step of plot_stats.py.

NO RE-RUNNING OF THE ACCUMULATION IS NEEDED

Usage
-----
    # Chemical species
    python plot_pointdensity.py --molecule CO --chemistry Crich --n-tasks 32

    # Physical parameters
    python plot_pointdensity.py --phys-param temperature --chemistry Crich --n-tasks 32
    python plot_pointdensity.py --phys-param av           --chemistry Crich --n-tasks 32
    python plot_pointdensity.py --phys-param density      --chemistry Crich --n-tasks 32

    # All three physical parameters in one go (single figure, 3 shared-x panels)
    python plot_pointdensity.py --phys-param all --chemistry Crich --n-tasks 32

Optional flags
--------------
    --overlay-stats / --no-overlay-stats
        Overplot the mean and 16th/84th percentile bands (default: on).

    --overlay-1d
        Overlay the 1-D chemistry model profile (molecule mode only).

    --overlay-particle PARTICLE_ID
        Overlay the radius-vs-value track of a single particle across all
        HDF5 dump files (e.g. --overlay-particle 2073529).

    --show
        Display the plot interactively.
"""
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

# ---------------------------------------------------------------------------
# Grid constants — MUST match plot_stats.py exactly
# ---------------------------------------------------------------------------

DUMP_DIR      = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
SCRATCH_DIR   = Path("/aphid/scratch-3month/sbeckers/v10a09_out/accum_scratch")
SAVE_DIR_BASE = BASE_PATH / "figures/v10a09_out"

R_MIN, R_MAX, N_BINS = 1e13, 1e17, 300
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

# ---------------------------------------------------------------------------
# Physical parameter config — MUST match plot_stats.py exactly
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 1-D model loader
# ---------------------------------------------------------------------------

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
    print(f"Loaded 1D model")
    return radius_1d, fracs_1d, mloss_label, vinf_label

# ---------------------------------------------------------------------------
# 1-D profile overlay
# ---------------------------------------------------------------------------

def overlay_1d_profile(
    ax,
    molecule:   str,
    chemistry:  str,
    color:      str   = "k",
    lw:         float = 2.0,
    ls:         str   = "-.",
    label:      str | None = None,
) -> None:
    """
    Overlay the 1-D chemistry model profile for *molecule* on *ax*.

    Parameters
    ----------
    ax        : matplotlib Axes
    molecule  : species name as it appears in the 1-D model output
    chemistry : "Crich" or "Orich"
    color     : line colour (default cyan for visibility on the plasma colourmap)
    lw        : line width
    ls        : line style
    label     : legend label; if None, auto-generates "1D (<mloss>, <vinf>)"
    """
    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)
    frac = get_fractional_abundance(fracs_1d, molecule)
    if frac is None:
        print(f"  overlay_1d_profile: '{molecule}' not found in 1-D model — skipping.")
        return
    _label = label if label is not None else f"1D ({mloss_label}, {vinf_label})"
    ax.plot(radius_1d, frac, lw=lw, ls=ls, color=color,
            label=_label, zorder=6)

# ---------------------------------------------------------------------------
# Single-particle track
# ---------------------------------------------------------------------------

def _normalise_species_key(name: str) -> str:
    """
    Convert a species name to its HDF5 dataset key under particles/.
    Mirrors plot_stats.normalise_name exactly so keys always match.
    """
    s = name.strip().lower()
    s = s.replace("+", "_plus").replace("-", "_minus").replace("/", "_")
    s = s.replace("(", "_").replace(")", "_").replace(".", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+",           "_", s).strip("_")
    return s


def load_particle_track(
    particle_id: int,
    molecule:    str | None = None,
    phys_param:  str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk every HDF5 dump file and collect the radius + value history of
    one particle, identified by its unique integer ID.

    The search uses ``np.where(ids == particle_id)`` on the
    ``particles/id`` array — no index structures required.

    Parameters
    ----------
    particle_id : int
        ID to search for (field: ``particles/id``).
    molecule : str or None
        Chemical species name.  The abundance is read from
        ``particles/<normalised_key>`` where the key is derived by
        ``_normalise_species_key(molecule)``, matching plot_stats.py.
        Mutually exclusive with *phys_param*.
    phys_param : str or None
        One of ``"temperature"``, ``"av"``, ``"density"``.  The value is
        read from ``particles/<hdf5_key>`` as defined in ``PHYS_PARAMS``.
        Mutually exclusive with *molecule*.

    Returns
    -------
    radii  : np.ndarray  — ``particles/r`` at each dump where the particle appears
    values : np.ndarray  — abundance or physical-parameter value at each dump

    Notes
    -----
    Results are sorted by radius (ascending) before being returned, so the
    track always runs left-to-right on a log-radius plot.
    """
    if (molecule is None) == (phys_param is None):
        raise ValueError("Provide exactly one of 'molecule' or 'phys_param'.")

    if phys_param is not None and phys_param not in PHYS_PARAMS:
        raise ValueError(
            f"Unknown phys_param '{phys_param}'. Choose from: {list(PHYS_PARAMS)}"
        )

    dump_files = sorted(DUMP_DIR.glob("*.h5"))
    if not dump_files:
        raise FileNotFoundError(f"No .hdf5 dump files found in {DUMP_DIR}")

    # Resolve the HDF5 field name once, before the loop
    if molecule is not None:
        value_field = f"particles/{_normalise_species_key(molecule)}"
    else:
        value_field = f"particles/{PHYS_PARAMS[phys_param]['hdf5_key']}"

    radii:  list[float] = []
    values: list[float] = []

    for fpath in dump_files:
        try:
            with h5py.File(fpath, "r") as f:
                ids = f["particles/id"][:]
                (match,) = np.where(ids == particle_id)

                if match.size == 0:
                    continue                          # particle not in this dump

                idx = int(match[0])
                radii.append( float(f["particles/r"][idx]) )
                values.append( float(f[value_field][idx])  )

        except Exception as exc:
            print(f"  load_particle_track: skipping {fpath.name}: {exc}")

    if not radii:
        raise ValueError(
            f"Particle {particle_id} was not found in any of the "
            f"{len(dump_files)} dump files under {DUMP_DIR}."
        )

    radii_arr  = np.array(radii,  dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)
    order      = np.argsort(radii_arr)

    print(
        f"  Particle {particle_id}: found in {len(radii_arr)} / "
        f"{len(dump_files)} dumps."
    )
    return radii_arr[order], values_arr[order]


def overlay_particle_track(
    ax,
    particle_id: int,
    molecule:    str | None = None,
    phys_param:  str | None = None,
    color:       str        = "black",
    lw:          float      = 1.5,
    ms:          float      = 4.0,
    label:       str | None = None,
) -> None:
    """
    Overlay the radius-vs-value track of one particle on *ax*.

    Drawn as a dashed line (``"--"``) with circular markers (``"o"``),
    keeping it visually distinct from the solid median and dotted mean
    lines already on the plot.

    Parameters
    ----------
    ax          : matplotlib Axes
    particle_id : int
    molecule    : species name  (mutually exclusive with phys_param)
    phys_param  : physical parameter key  (mutually exclusive with molecule)
    color       : defaults to ``"black"``
    lw          : line width
    ms          : marker size
    label       : legend entry; defaults to ``"Particle <id>"``
    """
    radii, values = load_particle_track(
        particle_id = particle_id,
        molecule    = molecule,
        phys_param  = phys_param,
    )

    ax.plot(
        radii, values,
        linestyle  = "--",
        marker     = "o",
        markersize = ms,
        linewidth  = lw,
        color      = color,
        label      = label if label is not None else f"Particle {particle_id}",
        zorder     = 7,           # above density map (pcm) and stats lines
    )

# ---------------------------------------------------------------------------
# Load & aggregate — chemical species
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stats overlay (generalised to arbitrary value-axis bins)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_density(
    hist_2d:        np.ndarray,
    val_edges:      np.ndarray,
    val_label:      str,
    bin_sum:        np.ndarray,
    bin_count:      np.ndarray,
    val_centres:    np.ndarray,
    title:          str | None   = None,
    param:          str | None   = None,
    clip_at_one:    bool         = False,
    apply_ab_limits: bool        = False,
    overlay_stats:  bool         = True,
    overlay_1d:     bool         = False,
    molecule:       str | None   = None,
    chemistry:      str          = "Crich",
    particle_id:    int | None   = None,
    vmin:           float | None = None,
    vmax:           float | None = None,
    save_path:      Path | None  = None,
    show:           bool         = False,
    dark_mode:      bool         = False,
) -> None:
    """
    Generic 2-D density plot on a log(r) × log(value) grid.

    Parameters
    ----------
    title          : axes title string (already formatted for LaTeX)
    hist_2d        : (N_BINS, n_val_bins) accumulated counts
    val_edges      : (n_val_bins + 1,) bin boundaries for the value axis
    val_label      : y-axis label
    bin_sum/count  : (N_BINS,) for the stats overlay
    val_centres    : (n_val_bins,) geometric centres for the stats overlay
    clip_at_one    : clip smoothed lines at 1 (True for abundances only)
    apply_ab_limits: call apply_abundance_axis_limits() (True for species)
    overlay_stats  : draw mean + percentile lines
    overlay_1d     : draw the 1-D chemistry model profile (molecule mode only)
    molecule       : species name — used by overlay_1d and overlay_particle
    chemistry      : "Crich" or "Orich"
    particle_id    : if set, overlay the track of this single particle
    vmin, vmax     : manual colour-scale limits
    save_path      : where to write the figure (None = don't save)
    show           : call plt.show()
    dark_mode      : use a dark background and colour scheme
    """
    h_float  = hist_2d.astype(np.float64)
    h_masked = np.ma.masked_where(h_float == 0, h_float)

    nonzero = h_float[h_float > 0]
    _vmin   = vmin if vmin is not None else nonzero.min()
    _vmax   = vmax if vmax is not None else nonzero.max()
    norm    = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # pcolormesh: X = R_EDGES, Y = val_edges, C = h.T
    # hist_2d shape is (N_BINS [r], n_val_bins [value]);
    # pcolormesh expects (value, r) so we transpose.
    pcm = ax.pcolormesh(
        R_EDGES, val_edges,
        h_masked.T,
        cmap="plasma",
        norm=norm,
        rasterized=True,
        shading="flat",
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Particle count per bin", fontsize=12)

    if overlay_stats:
        stats = compute_stats(
            bin_sum, bin_count, hist_2d,
            val_centres=val_centres,
            clip_at_one=clip_at_one,
        )
        primary_c_white = "white" if dark_mode else "k"
        primary_c_k     = "k"     if dark_mode else "white"
        ax.plot(R_CENTRES, stats["mean"], lw=1.5, ls=":",
                color=primary_c_white, alpha=0.8, label="Mean", zorder=5)
        ax.plot(R_CENTRES, stats["p50"], lw=2.0, ls="-",
                color=primary_c_white, label="Median", zorder=5)

    if overlay_1d and molecule is not None:
        overlay_1d_profile(ax, molecule=molecule, chemistry=chemistry)

    if particle_id is not None:
        overlay_particle_track(
            ax,
            particle_id = particle_id,
            molecule    = molecule,
            phys_param  = param,
        )

    # Single legend call after all overlays so entries are always complete
    if overlay_stats or overlay_1d or particle_id is not None:
        primary_c_white = "white" if dark_mode else "k"
        primary_c_k     = "k"     if dark_mode else "white"
        ax.legend(loc="lower left", fontsize=12,
                  framealpha=0.6, labelcolor=primary_c_white,
                  facecolor=primary_c_k)

    ax.set_xscale("log")
    ax.set_yscale("log")
    if apply_ab_limits:
        apply_abundance_axis_limits(ax)
        occupied_r = np.where(hist_2d.sum(axis=1) > 0)[0]
        if len(occupied_r) > 0:
            rmin = R_EDGES[occupied_r[0]]
            rmax = R_EDGES[occupied_r[-1] + 1]
            ax.set_xlim(rmin, rmax)
    else:
        all_x, all_y = _collect_xy([ax])
        ax.set_xlim(min(all_x), max(all_x))
        if param == "temperature":
            ax.set_ylim(min(all_y), max(all_y) * 4)
        elif param == "av":
            ax.set_ylim(min(all_y), max(all_y) * 2)
        elif param == "density":
            ax.set_ylim(min(all_y), max(all_y) * 10)
    add_log_ticks(ax)

    ax.set_xlabel("Radius [cm]", fontsize=13)
    ax.set_ylabel(val_label, fontsize=13)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Combined 3-panel figure for all physical parameters (shared x-axis)
# ---------------------------------------------------------------------------

def plot_density_all_phys(
    data_per_param: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    overlay_stats:  bool         = True,
    particle_id:    int | None   = None,
    vmin:           float | None = None,
    vmax:           float | None = None,
    save_path:      Path | None  = None,
    show:           bool         = False,
    dark_mode:      bool         = False,
) -> None:
    """
    Plot all three physical parameters in a single figure with three
    vertically stacked panels that share the x (radius) axis.

    Parameters
    ----------
    data_per_param : dict mapping each param name to the
                     (bin_sum, bin_count, hist_2d) tuple returned by
                     aggregate_phys_param().
    overlay_stats  : draw mean + median lines on every panel
    particle_id    : if set, overlay the single-particle track on every panel
    All other arguments have the same meaning as in plot_density().
    """
    params = list(PHYS_PARAMS)   # ["temperature", "av", "density"]
    n      = len(params)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(9, 6 * n / 2.2),
        sharex=True,
        dpi=300,
    )

    headroom = {"temperature": 4, "av": 2, "density": 10}

    for ax, param in zip(axes, params):
        cfg                         = PHYS_PARAMS[param]
        bin_sum, bin_count, hist_2d = data_per_param[param]

        h_float  = hist_2d.astype(np.float64)
        h_masked = np.ma.masked_where(h_float == 0, h_float)

        nonzero = h_float[h_float > 0]
        _vmin   = vmin if vmin is not None else nonzero.min()
        _vmax   = vmax if vmax is not None else nonzero.max()
        norm    = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)

        pcm = ax.pcolormesh(
            R_EDGES, cfg["edges"],
            h_masked.T,
            cmap="plasma",
            norm=norm,
            rasterized=True,
            shading="flat",
        )
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label("Particle count per bin", fontsize=10)

        if overlay_stats:
            stats = compute_stats(
                bin_sum, bin_count, hist_2d,
                val_centres=cfg["centres"],
                clip_at_one=False,
            )
            primary_c_white = "white" if dark_mode else "k"
            primary_c_k     = "k"     if dark_mode else "white"
            ax.plot(R_CENTRES, stats["mean"], lw=1.5, ls=":",
                    color=primary_c_white, alpha=0.8, label="Mean", zorder=5)
            ax.plot(R_CENTRES, stats["p50"], lw=2.0, ls="-",
                    color=primary_c_white, label="Median", zorder=5)

        if particle_id is not None:
            overlay_particle_track(
                ax,
                particle_id = particle_id,
                phys_param  = param,
            )

        # Single legend call after all overlays
        if overlay_stats or particle_id is not None:
            primary_c_white = "white" if dark_mode else "k"
            primary_c_k     = "k"     if dark_mode else "white"
            ax.legend(loc="lower left", fontsize=10,
                      framealpha=0.6, labelcolor=primary_c_white,
                      facecolor=primary_c_k)

        ax.set_xscale("log")
        ax.set_yscale("log")

        all_x, all_y = _collect_xy([ax])
        ax.set_xlim(min(all_x), max(all_x))
        ax.set_ylim(min(all_y), max(all_y) * headroom[param])

        add_log_ticks(ax)
        ax.set_ylabel(cfg["label"], fontsize=12)

    # x-label only on the bottom panel
    axes[-1].set_xlabel("Radius [cm]", fontsize=13)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


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
        description=(
            "Plot 2-D particle density (radius × value) for a molecule or "
            "physical parameter using pre-accumulated histogram data from "
            "plot_stats.py."
        )
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--molecule",
        nargs="+",
        type=str,
        help="Chemical species to plot (e.g. CO HCl SiO).",
    )
    target.add_argument(
        "--phys-param",
        choices=[*PHYS_PARAMS, "all"],
        help=(
            "Physical parameter to plot: 'temperature', 'av', 'density', "
            "or 'all' to produce a single figure with all three panels."
        ),
    )

    parser.add_argument("--chemistry", choices=["Crich", "Orich"],
                        default="Crich")
    parser.add_argument(
        "--n-tasks", type=int, default=1,
        help="Must match the n-tasks used in the accumulate step.",
    )
    parser.add_argument(
        "--overlay-stats", action="store_true", default=True,
        help="Overplot mean + median lines (default: on).",
    )
    parser.add_argument(
        "--no-overlay-stats", dest="overlay_stats", action="store_false",
    )
    parser.add_argument(
        "--overlay-1d", action="store_true", default=False,
        help="Overlay the 1-D chemistry model profile (molecule mode only).",
    )
    parser.add_argument(
        "--overlay-particle",
        type    = int,
        default = None,
        metavar = "PARTICLE_ID",
        help    = (
            "Overlay the radius-vs-value track of a single particle "
            "(e.g. --overlay-particle 2073529)."
        ),
    )
    parser.add_argument("--vmin", type=float, default=None,
                        help="Manual colour-scale lower limit.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Manual colour-scale upper limit.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dark-mode", action="store_true")

    mol_list = ['CO', 'CH2', 'CH3', 'CH4', 'HCl', 'CH3CN', 'SiO', 'HCN', 
                'CN', 'HC3N', 'HC5N', 'HC7N', 'C2H', 'C4H', 'C6H', 'SiC', 
                'SiN', 'H2CS', 'H2CO', 'N2', 'NH3', 'H2S', 'HCP', 'H2O', 'C2H2', 
                'CS', 'SiC2', 'HF', 'C2H4', 'SiS']
    
    if is_interactive():
        args = parser.parse_args([
            "--molecule",       *mol_list,
            "--chemistry",      "Crich",
            "--n-tasks",        "32",
            "--overlay-stats",
            "--overlay-1d",
            # "--overlay-particle", "2073529",
            # "--show",
        ])
    else:
        args = parser.parse_args()

    save_dir = SAVE_DIR_BASE / args.chemistry / "point_density"
    save_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style(args.dark_mode)

    save_ext = "pdf" if args.dark_mode else "png"

    if args.molecule:
        for molecule in args.molecule:
            print(f"\nProcessing molecule: {molecule}")

            bin_sum, bin_count, hist_2d = aggregate_molecule(
                molecule, args.chemistry, args.n_tasks
            )
            plot_density(
                title            = rf"{format_species_label(molecule)}",
                hist_2d          = hist_2d,
                val_edges        = AB_EDGES,
                val_label        = r"Abundance relative to $\mathrm{H}_2$",
                bin_sum          = bin_sum,
                bin_count        = bin_count,
                val_centres      = AB_CENTRES,
                clip_at_one      = True,
                apply_ab_limits  = True,
                overlay_stats    = args.overlay_stats,
                overlay_1d       = args.overlay_1d,
                molecule         = molecule,
                chemistry        = args.chemistry,
                particle_id      = args.overlay_particle,
                vmin             = args.vmin,
                vmax             = args.vmax,
                save_path        = save_dir / f"point_density_{molecule}.{save_ext}",
                show             = args.show,
                dark_mode        = args.dark_mode,
            )
    else:
        if args.phys_param == "all":
            data_per_param = {}
            for param in PHYS_PARAMS:
                data_per_param[param] = aggregate_phys_param(
                    param, args.chemistry, args.n_tasks
                )
            plot_density_all_phys(
                data_per_param = data_per_param,
                overlay_stats  = args.overlay_stats,
                particle_id    = args.overlay_particle,
                vmin           = args.vmin,
                vmax           = args.vmax,
                save_path      = save_dir / f"point_density_phys_all.{save_ext}",
                show           = args.show,
                dark_mode      = args.dark_mode,
            )
        else:
            param = args.phys_param
            bin_sum, bin_count, hist_2d = aggregate_phys_param(
                param, args.chemistry, args.n_tasks
            )
            cfg = PHYS_PARAMS[param]
            plot_density(
                hist_2d          = hist_2d,
                val_edges        = cfg["edges"],
                val_label        = cfg["label"],
                bin_sum          = bin_sum,
                bin_count        = bin_count,
                val_centres      = cfg["centres"],
                title            = None,
                param            = param,
                clip_at_one      = False,
                apply_ab_limits  = False,
                overlay_stats    = args.overlay_stats,
                particle_id      = args.overlay_particle,
                vmin             = args.vmin,
                vmax             = args.vmax,
                save_path        = save_dir / f"point_density_{param}.{save_ext}",
                show             = args.show,
                dark_mode        = args.dark_mode,
            )


if __name__ == "__main__":
    main()