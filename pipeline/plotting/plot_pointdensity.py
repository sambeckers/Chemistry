#!/usr/bin/env python3
"""
plot_pointdensity.py

Visualises the 2-D particle density (radius × value) for a single molecule
OR a physical parameter (temperature, AV, density) by reading the partial
.npz files produced by the accumulate step of plot_stats.py.

NO RE-RUNNING OF THE ACCUMULATION IS NEEDED — this script is purely a
post-processing visualisation step that reuses what is already on disk.

Usage
-----
    # Chemical species
    python plot_pointdensity.py --molecule CO --chemistry Crich --n-tasks 32

    # Physical parameters
    python plot_pointdensity.py --phys-param temperature --chemistry Crich --n-tasks 32
    python plot_pointdensity.py --phys-param av           --chemistry Crich --n-tasks 32
    python plot_pointdensity.py --phys-param density      --chemistry Crich --n-tasks 32

    # All three physical parameters in one go
    python plot_pointdensity.py --phys-param all --chemistry Crich --n-tasks 32

Optional flags
--------------
    --overlay-stats / --no-overlay-stats
        Overplot the mean and 16th/84th percentile bands (default: on).

    --vmin, --vmax
        Manual colour-scale limits. Useful when a handful of very dense
        bins wash out the rest.

    --show
        Display the plot interactively.

Why pcolormesh instead of hexbin?
----------------------------------
matplotlib's hexbin() requires all raw (x, y) pairs in memory.  With
~6 billion data points that is not feasible.  The partial .npz files
already contain a 2-D histogram H[r_bin, val_bin] accumulated in the same
log-spaced grids used throughout this project.  Plotting that histogram
with pcolormesh gives the same visual information as hexbin — colour
encodes local density — while respecting the log-spaced geometry of the
bins.  Hexagonal tiling would actually be *worse* here because our optimal
bin boundaries already account for the logarithmic axes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica",
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white",
})

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_PATH
from code_chem.plotting.plot_utils import apply_abundance_axis_limits, add_log_ticks

# ---------------------------------------------------------------------------
# Grid constants — MUST match plot_stats.py exactly
# ---------------------------------------------------------------------------

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
        "log_min":  0.0,    # 1 K
        "log_max":  6.0,    # 1 000 000 K
        "n_bins":   150,
        "label":    "Temperature [K]",
        "color":    "#d6604d",
    },
    "av": {
        "log_min": -6.0,    # 10^-6 mag
        "log_max":  1.0,    # 10 mag
        "n_bins":   150,
        "label":    r"$A_V$ [mag]",
        "color":    "#4dac26",
    },
    "density": {
        "log_min": -1.0,    # 0.1 cm^-3
        "log_max": 10.0,    # 10^10 cm^-3
        "n_bins":  150,
        "label":   r"Density [$n$, cm$^{-3}$]",
        "color":   "#762a83",
    },
}

for _cfg in PHYS_PARAMS.values():
    _cfg["edges"]   = np.logspace(_cfg["log_min"], _cfg["log_max"],
                                  _cfg["n_bins"] + 1)
    _cfg["centres"] = np.sqrt(_cfg["edges"][:-1] * _cfg["edges"][1:])


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
# Core plot function (shared by species and physical parameters)
# ---------------------------------------------------------------------------

def plot_density(
    title:          str,
    hist_2d:        np.ndarray,
    val_edges:      np.ndarray,
    val_label:      str,
    bin_sum:        np.ndarray,
    bin_count:      np.ndarray,
    val_centres:    np.ndarray,
    clip_at_one:    bool        = False,
    apply_ab_limits: bool       = False,
    overlay_stats:  bool        = True,
    vmin:           float | None = None,
    vmax:           float | None = None,
    save_path:      Path | None  = None,
    show:           bool         = False,
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
    vmin, vmax     : manual colour-scale limits
    save_path      : where to write the figure (None = don't save)
    show           : call plt.show()
    """
    h_float  = hist_2d.astype(np.float64)
    h_masked = np.ma.masked_where(h_float == 0, h_float)

    nonzero = h_float[h_float > 0]
    _vmin   = vmin if vmin is not None else nonzero.min()
    _vmax   = vmax if vmax is not None else nonzero.max()
    norm    = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=600)

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
        ax.plot(R_CENTRES, stats["mean"], lw=2.0, ls="-",
                color="white", label="Mean", zorder=5)
        ax.plot(R_CENTRES, stats["p50"], lw=1.5, ls=":",
                color="white", alpha=0.8, label="Median", zorder=5)
        ax.fill_between(
            R_CENTRES, stats["p16"], stats["p84"],
            color="white", alpha=0.20, zorder=4,
            label=r"16th–84th pct.",
        )
        ax.legend(loc="lower left", fontsize=12,
                  framealpha=0.6, labelcolor="white",
                  facecolor="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    if apply_ab_limits:
        apply_abundance_axis_limits(ax)
    add_log_ticks(ax)

    ax.set_xlabel("Radius [cm]", fontsize=13)
    ax.set_ylabel(val_label, fontsize=13)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def plot_molecule_density(
    molecule:      str,
    chemistry:     str,
    bin_sum:       np.ndarray,
    bin_count:     np.ndarray,
    hist_2d:       np.ndarray,
    overlay_stats: bool        = True,
    vmin:          float | None = None,
    vmax:          float | None = None,
    save_path:     Path | None  = None,
    show:          bool         = False,
) -> None:
    """Density plot for a chemical species (abundance axis)."""
    plot_density(
        title           = rf"\textbf{{{molecule}}} — point density",
        hist_2d         = hist_2d,
        val_edges       = AB_EDGES,
        val_label       = r"Abundance (wrt H$_{\mathrm{nuc}}$)",
        bin_sum         = bin_sum,
        bin_count       = bin_count,
        val_centres     = AB_CENTRES,
        clip_at_one     = True,
        apply_ab_limits = True,
        overlay_stats   = overlay_stats,
        vmin            = vmin,
        vmax            = vmax,
        save_path       = save_path,
        show            = show,
    )


def plot_phys_density(
    param:         str,
    bin_sum:       np.ndarray,
    bin_count:     np.ndarray,
    hist_2d:       np.ndarray,
    overlay_stats: bool        = True,
    vmin:          float | None = None,
    vmax:          float | None = None,
    save_path:     Path | None  = None,
    show:          bool         = False,
) -> None:
    """Density plot for a physical parameter (param-specific value axis)."""
    cfg = PHYS_PARAMS[param]
    plot_density(
        title           = rf"\textbf{{{param}}} — point density",
        hist_2d         = hist_2d,
        val_edges       = cfg["edges"],
        val_label       = cfg["label"],
        bin_sum         = bin_sum,
        bin_count       = bin_count,
        val_centres     = cfg["centres"],
        clip_at_one     = False,   # physical params have no ≤ 1 constraint
        apply_ab_limits = False,
        overlay_stats   = overlay_stats,
        vmin            = vmin,
        vmax            = vmax,
        save_path       = save_path,
        show            = show,
    )


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
        "--molecule", type=str,
        help="Chemical species to plot (e.g. CO, HCl).",
    )
    target.add_argument(
        "--phys-param",
        choices=[*PHYS_PARAMS, "all"],
        help=(
            "Physical parameter to plot: 'temperature', 'av', 'density', "
            "or 'all' to produce a separate figure for each."
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
        help="Overplot mean + 16th/84th percentile lines (default: on).",
    )
    parser.add_argument(
        "--no-overlay-stats", dest="overlay_stats", action="store_false",
    )
    parser.add_argument("--vmin", type=float, default=None,
                        help="Manual colour-scale lower limit.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Manual colour-scale upper limit.")
    parser.add_argument("--show", action="store_true")

    if is_interactive():
        args = parser.parse_args([
            "--phys-param",     "all",
            "--chemistry",      "Crich",
            "--n-tasks",        "32",
            "--overlay-stats",
            "--show",
        ])
    else:
        args = parser.parse_args()

    save_dir = SAVE_DIR_BASE / args.chemistry
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.molecule:
        bin_sum, bin_count, hist_2d = aggregate_molecule(
            args.molecule, args.chemistry, args.n_tasks
        )
        plot_molecule_density(
            molecule      = args.molecule,
            chemistry     = args.chemistry,
            bin_sum       = bin_sum,
            bin_count     = bin_count,
            hist_2d       = hist_2d,
            overlay_stats = args.overlay_stats,
            vmin          = args.vmin,
            vmax          = args.vmax,
            save_path     = save_dir / f"particle_density_{args.molecule}.pdf",
            show          = args.show,
        )

    else:
        # Physical parameter(s)
        params = list(PHYS_PARAMS) if args.phys_param == "all" else [args.phys_param]
        for param in params:
            bin_sum, bin_count, hist_2d = aggregate_phys_param(
                param, args.chemistry, args.n_tasks
            )
            plot_phys_density(
                param         = param,
                bin_sum       = bin_sum,
                bin_count     = bin_count,
                hist_2d       = hist_2d,
                overlay_stats = args.overlay_stats,
                vmin          = args.vmin,
                vmax          = args.vmax,
                save_path     = save_dir / f"particle_density_{param}.pdf",
                show          = args.show,
            )


if __name__ == "__main__":
    main()