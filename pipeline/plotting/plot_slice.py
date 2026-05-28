#!/usr/bin/env python3
"""
plot_slice.py

Plots a 2-D spatial slice (xy or xz plane) of any scalar field — or a
product of two fields, e.g. n × CO fractional abundance — either from a
single HDF5 snapshot

Single-dump mode  (default, reads one snapshot directly)
---------------------------------------------------------
    # List all available dump files with indices
    python plot_slice.py --list-dumps

    # xy-slice of gas density from dump index 42
    python plot_slice.py --dump-index 42 --plane xy --quantity density

    # xz-slice of n_CO  (density × CO fractional abundance)
    python plot_slice.py --dump-index 42 --plane xz \
           --quantity density --times CO

    # Explicit path, custom slice thickness and number of bins
    python plot_slice.py --dump /path/to/dump_0042.h5 \
           --plane xy --quantity CO --slice-frac 0.05 --n-bins 256

    # Use only 10% of particles, interpolate gaps
    python plot_slice.py --dump-index 42 --plane xy --quantity density \
           --fraction 0.1 --interpolate

    # Side-by-side comparison of 10%, 50%, 100% subsampling
    python plot_slice.py --dump-index 42 --plane xy --quantity density \
           --compare-fractions --interpolate

    # Plot slices for multiple molecules in one go
    python plot_slice.py --dump-index 42 --plane xz \
           --quantity CO CH4 SiO --compare-fractions --interpolate

Optional flags
--------------
    --quantity Q [Q ...]      One or more field names to plot (loops over each)
    --fraction F              Keep only fraction F of particles (0-1, default 1.0)
    --interpolate             Fill empty cells by nearest-neighbour interpolation
    --compare-fractions       Plot 10%, 50%, 100% side-by-side in one figure
    --log / --no-log          Log colour scale (default: on)
    --cmap NAME               Matplotlib colourmap (default: plasma)
    --vmin / --vmax           Manual colour limits
    --coord-lim L             Manual half-extent for both axes [cm]
    --show                    Display interactively
    --dark-mode               Dark background
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_PATH
from code_chem.plotting.plot_utils import set_plot_style

# ---------------------------------------------------------------------------
# Paths  — must match plot_stats.py
# ---------------------------------------------------------------------------

DUMP_DIR      = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
SCRATCH_DIR   = Path("/aphid/scratch-3month/sbeckers/v10a09_out/accum_scratch")
SAVE_DIR_BASE = BASE_PATH / "figures/v10a09_out"

# ---------------------------------------------------------------------------
# Spatial grid defaults
# ---------------------------------------------------------------------------

N_SLICE_BINS  = 512     # bins per spatial axis (single-dump mode)
MIN_PER_CELL  = 1       # cells with fewer particles are shown as NaN

# Physical-parameter → HDF5 dataset key  (mirrors plot_stats.py)
PHYS_HDF5_KEY: dict[str, str] = {
    "density":     "density",
    "temperature": "temp",
    "av":          "av",
}

# ---------------------------------------------------------------------------
# Deterministic particle subsampling  (mirrors plot_stats.py)
# ---------------------------------------------------------------------------

# Floor of 2^64 divided by the golden ratio — maps sequential integers
# uniformly across [0, 2^64).  See Knuth TAOCP Vol. 3 §6.4.
_KNUTH_CONST = np.uint64(0x9E3779B97F4A7C15)


def _particle_keep_mask(
    ids: np.ndarray,
    fraction: float,
    seed: int = 123456789,
) -> np.ndarray:
    """
    Return a boolean mask selecting a deterministic ~*fraction* of particles.

    Identical to the implementation in plot_stats.py so that the same
    particle IDs are selected when both scripts process the same dump.
    """
    if fraction >= 1.0 - 1e-9:
        return np.ones(len(ids), dtype=bool)
    if fraction <= 1e-9:
        return np.zeros(len(ids), dtype=bool)
    h = (ids.astype(np.uint64) ^ np.uint64(seed)) * _KNUTH_CONST
    return h < np.uint64(int(fraction * 0xFFFF_FFFF_FFFF_FFFF))


# ---------------------------------------------------------------------------
# NaN interpolation
# ---------------------------------------------------------------------------

def fill_nans_nearest(
    arr:          np.ndarray,
    max_distance: float = 5.0,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    """
    Fill NaN cells near the particle boundary using NearestNDInterpolator,
    then apply a Gaussian blur in log-space to soften the choppy edges.

    Parameters
    ----------
    arr          : (M, N) float array, may contain NaN
    max_distance : cells further than this many grid pixels from the nearest
                   populated cell are left as NaN (default: 5).
                   Prevents the empty outer background from being filled.
    smooth_sigma : standard deviation (grid pixels) of the log-space Gaussian
                   applied after filling (default: 3.0).  Set to 0 to skip.
                   Smoothing is done in log-space so the many-orders-of-
                   magnitude dynamic range is handled uniformly; it uses a
                   weighted average so remaining NaN cells don't bleed into
                   valid data.

    Returns
    -------
    Filled and smoothed copy of *arr*.  Cells beyond *max_distance* of any
    populated cell remain NaN.
    """
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr
    if nan_mask.all():
        return arr

    # ---- 1. Identify cells to fill (within max_distance of a valid cell) ----
    distances = distance_transform_edt(nan_mask)
    close_mask = nan_mask & (distances <= max_distance)

    if close_mask.any():
        # Build interpolator from all valid cells
        valid_rows, valid_cols = np.where(~nan_mask)
        interp = NearestNDInterpolator(
            np.column_stack([valid_rows, valid_cols]),
            arr[valid_rows, valid_cols],
        )
        fill_rows, fill_cols = np.where(close_mask)
        filled = arr.copy()
        filled[fill_rows, fill_cols] = interp(fill_rows, fill_cols)
    else:
        filled = arr.copy()

    # ---- 2. Log-space Gaussian smoothing ----
    if smooth_sigma > 0:
        has_data = np.isfinite(filled) & (filled > 0)
        log_arr  = np.where(has_data, np.log10(filled), 0.0)

        # Weighted blur so NaN regions don't drag down valid edge cells
        blurred_log    = gaussian_filter(log_arr,                sigma=smooth_sigma)
        blurred_weight = gaussian_filter(has_data.astype(float), sigma=smooth_sigma)

        with np.errstate(invalid="ignore", divide="ignore"):
            log_smooth = np.where(
                blurred_weight > 0,
                blurred_log / blurred_weight,
                np.nan,
            )
        filled = np.where(has_data, 10.0 ** log_smooth, filled)

    return filled


# ---------------------------------------------------------------------------
# Name normalisation  (mirrors plot_stats.py)
# ---------------------------------------------------------------------------

def normalise_name(name: str) -> str:
    """Map a species/molecule name to its HDF5 dataset key."""
    c = name.strip().lower()
    c = c.replace("+", "_plus").replace("-", "_minus").replace("/", "_")
    c = c.replace("(", "_").replace(")", "_").replace(".", "_")
    c = re.sub(r"[^a-z0-9_]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


# ---------------------------------------------------------------------------
# HDF5 readers
# ---------------------------------------------------------------------------

def read_coords(f: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read (x, y, z) particle coordinates from an open HDF5 file.

    Tries the following storage conventions in order:
      1. particles/x, particles/y, particles/z   — separate 1-D arrays
      2. particles/pos, particles/xyz, particles/coordinates, particles/coords
         — packed (N, 3) array, columns [x, y, z]
      3. Fallback: x = particles/r,  y = z = 0   (warns if used)
    """
    p = f["particles"]

    if all(k in p for k in ("x", "y", "z")):
        return p["x"][:], p["y"][:], p["z"][:]

    for key in ("pos", "xyz", "coordinates", "coords"):
        if key in p:
            arr = p[key][:]
            return arr[:, 0], arr[:, 1], arr[:, 2]

    print(
        "  [WARNING] No x/y/z or packed position array found in 'particles'.\n"
        "  Using r as x and zeros for y, z.  Slice plots will be 1-D."
    )
    r = p["r"][:]
    return r, np.zeros_like(r), np.zeros_like(r)


def read_field(f: h5py.File, name: str) -> np.ndarray:
    """
    Read a scalar field from the 'particles' group of an open HDF5 file.

    Accepts physical parameter shortcuts ('density', 'temperature', 'av')
    and chemical species names ('CO', 'HCl', 'SiO', …).
    """
    p = f["particles"]

    if name in PHYS_HDF5_KEY:
        key = PHYS_HDF5_KEY[name]
        if key in p:
            return p[key][:]
        raise KeyError(
            f"Physical parameter '{name}' → HDF5 key '{key}' not found in dump.\n"
            f"Available keys: {sorted(p.keys())}"
        )

    key = normalise_name(name)
    if key in p:
        return p[key][:]
    raise KeyError(
        f"Species '{name}' → normalised key '{key}' not found in dump.\n"
        f"Available keys: {sorted(p.keys())}"
    )


# ---------------------------------------------------------------------------
# Dump discovery
# ---------------------------------------------------------------------------

def get_all_dumps(dump_dir: Path = DUMP_DIR) -> list[Path]:
    return sorted(dump_dir.glob("dump_*.h5"))


def resolve_dump(
    dump_dir:   Path,
    dump_path:  str | None,
    dump_index: int | None,
) -> Path:
    if dump_path is not None:
        p = Path(dump_path)
        if not p.exists():
            raise FileNotFoundError(f"Dump file not found: {p}")
        return p

    dumps = get_all_dumps(dump_dir)
    if not dumps:
        raise FileNotFoundError(f"No dump_*.h5 files found in {dump_dir}")
    if dump_index is None:
        raise ValueError("Provide --dump <path> or --dump-index <N>.")
    if not (-len(dumps) <= dump_index < len(dumps)):
        raise IndexError(
            f"--dump-index {dump_index} is out of range "
            f"(valid: 0 – {len(dumps) - 1}, or negative from end)."
        )
    return dumps[dump_index]


# ---------------------------------------------------------------------------
# Core: build a 2-D slice map from a single dump
# ---------------------------------------------------------------------------

def _field_mask(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Return a boolean validity mask for a field array, mirroring the masks
    applied in plot_stats.py's _accumulate_dump:
      • physical parameters (density, temperature, av): finite & > 0
      • chemical species / abundances:                  finite & > 0 & <= 1
    """
    base = np.isfinite(arr) & (arr > 0)
    if name in PHYS_HDF5_KEY:
        return base
    return base & (arr <= 1.0)


def build_slice_map(
    dump_path:   Path,
    plane:       str,
    quantity:    str,
    times:       str | None,
    n_bins:      int   = N_SLICE_BINS,
    coord_lim:   float | None = None,
    fraction:    float = 1.0,
    interpolate: bool  = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Read *dump_path*, project particles onto a 2-D grid, and return
    the mean quantity per spatial cell.

    Parameters
    ----------
    fraction    : keep only this fraction of particles (via deterministic
                  ID-based hash, identical to plot_stats.py).  fraction=1.0
                  keeps all particles and skips the ID read entirely.
    interpolate : if True, fill empty cells by nearest-neighbour
                  interpolation before returning.

    Returns
    -------
    xedges, yedges, mean_map, count_map, meta
    """
    if plane not in ("xy", "xz"):
        raise ValueError(f"plane must be 'xy' or 'xz', got '{plane!r}'")

    frac_str = f"{int(fraction * 100)}%" if fraction < 1.0 - 1e-9 else "100%"
    print(f"\nReading {dump_path.name}  (fraction={frac_str}) …")

    with h5py.File(dump_path, "r") as f:
        x_all, y_all, z_all = read_coords(f)
        q1 = read_field(f, quantity)
        q2 = read_field(f, times) if times is not None else None

        # Only read particle IDs when actually subsampling
        if fraction < 1.0 - 1e-9:
            keep_mask = _particle_keep_mask(f["particles/id"][:], fraction)
        else:
            keep_mask = None

    if quantity == 'density':
        q_label = rf"n{times} [m$^{{-3}}$]" if times is not None else r"$n$ [m$^{-3}$]"
    elif quantity == 'temperature':
        q_label = rf"$T${times} [K]" if times is not None else r"$T$ [K]"
    elif quantity == 'av':
        q_label = rf"$A_V${times} [mag]" if times is not None else r"$A_V$ [mag]"
    else:
        q_label = rf"{quantity} abundance (wrt H$_{{\mathrm{{nuc}}}}$)"

    if plane == "xy":
        ah, av_arr = x_all, y_all
        lh, lv     = "x [cm]", "y [cm]"
    else:
        ah, av_arr = x_all, z_all
        lh, lv     = "x [cm]", "z [cm]"

    # Combine validity mask with optional particle-fraction mask
    mask = _field_mask(q1, quantity)
    if q2 is not None:
        mask &= _field_mask(q2, times)
    if keep_mask is not None:
        mask &= keep_mask

    ah_v = ah[mask]
    av_v = av_arr[mask]
    q_v  = q1[mask] * q2[mask] if q2 is not None else q1[mask]

    n_total = len(x_all)
    n_valid = int(mask.sum())
    print(f"  {n_valid:,} / {n_total:,} particles pass the validity + fraction mask")

    if n_valid == 0:
        raise RuntimeError(
            "No valid particles remain after the validity mask.\n"
            "Check the field name and that the dump contains the requested species."
        )

    _clim  = coord_lim if coord_lim is not None else (
        max(np.abs(ah_v).max(), np.abs(av_v).max()) * 1.05
    )
    xedges = np.linspace(-_clim, _clim, n_bins + 1)
    yedges = np.linspace(-_clim, _clim, n_bins + 1)

    count_map, _, _ = np.histogram2d(ah_v, av_v, bins=[xedges, yedges])
    sum_map,   _, _ = np.histogram2d(ah_v, av_v, bins=[xedges, yedges],
                                     weights=q_v)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_map = np.where(
            count_map >= MIN_PER_CELL,
            sum_map / count_map,
            np.nan,
        )

    n_empty = int(np.isnan(mean_map).sum())
    n_cells = n_bins * n_bins
    do_interpolate = interpolate and fraction < 1.0 - 1e-9
    if n_empty > 0:
        print(f"  {n_empty:,} / {n_cells:,} cells empty "
              f"({100 * n_empty / n_cells:.1f}%)"
              + (" \u2192 interpolating" if do_interpolate else ""))

    if do_interpolate:
        mean_map = fill_nans_nearest(mean_map)

    # dump_name: numeric part only, e.g. "01600" from "dump_01600.h5"
    dump_name = dump_path.stem.split("_", 1)[1]

    meta = {
        "h_label":    lh,
        "v_label":    lv,
        "q_label":    q_label,
        "dump_name":  dump_name,
        "n_total":    n_total,
        "n_valid":    n_valid,
        "plane":      plane,
        "fraction":   fraction,
        "interpolate": do_interpolate,
    }
    return xedges, yedges, mean_map, count_map, meta


# ---------------------------------------------------------------------------
# Plot — single panel
# ---------------------------------------------------------------------------

def plot_slice_map(
    xedges:    np.ndarray,
    yedges:    np.ndarray,
    mean_map:  np.ndarray,
    meta:      dict,
    quantity:  str,
    log_scale: bool         = True,
    cmap:      str          = "inferno",
    vmin:      float | None = None,
    vmax:      float | None = None,
    save_path: Path | None  = None,
    show:      bool         = False,
    dark_mode: bool         = False,
) -> None:
    """Render a single 2-D slice map with pcolormesh."""
    cm_to_kAU = (1 * u.cm).to(u.au).value / 1e3
    cm3_to_m3 = (1 * u.cm**-3).to(u.m**-3).value

    xedges_kAU = xedges * cm_to_kAU
    yedges_kAU = yedges * cm_to_kAU

    data = mean_map.T * cm3_to_m3 if quantity == "density" else mean_map.T

    finite_pos = data[np.isfinite(data) & (data > 0)]
    if len(finite_pos) == 0:
        print("  Warning: no finite positive values to plot — figure will be empty.")
        return

    _vmin = vmin if vmin is not None else finite_pos.min()
    _vmax = vmax if vmax is not None else finite_pos.max()
    norm  = (mcolors.LogNorm(vmin=_vmin, vmax=_vmax) if log_scale
             else mcolors.Normalize(vmin=_vmin, vmax=_vmax))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    pcm = ax.pcolormesh(
        xedges_kAU, yedges_kAU,
        np.ma.masked_invalid(data),
        cmap=cmap, norm=norm, rasterized=True, shading="flat",
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(meta["q_label"], fontsize=12)

    ax.set_xlabel(meta["h_label"].replace("[cm]", r"[$10^3$ AU]"), fontsize=13)
    ax.set_ylabel(meta["v_label"].replace("[cm]", r"[$10^3$ AU]"), fontsize=13)

    frac_pct = int(meta["fraction"] * 100)
    interp_tag = " (interpolated)" if meta["interpolate"] else ""
    ax.set_title(
        rf"Dump {meta['dump_name']} — {frac_pct}\% of particles{interp_tag}",
        fontsize=13,
    )
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.ticklabel_format(style="plain")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — 1×3 fraction comparison
# ---------------------------------------------------------------------------

def plot_fraction_compare(
    dump_path:   Path,
    plane:       str,
    quantity:    str,
    times:       str | None,
    n_bins:      int,
    coord_lim:   float | None,
    interpolate: bool,
    log_scale:   bool,
    cmap:        str,
    vmin:        float | None,
    vmax:        float | None,
    save_path:   Path | None = None,
    show:        bool        = False,
    dark_mode:   bool        = False,
) -> None:
    """
    Build slice maps at 10%, 50%, and 100% particle subsampling and display
    them side-by-side in a 1×3 figure with a shared colour scale.

    The colour scale is derived from the 100% map (ground truth), so that
    subsampled panels are directly comparable without being skewed by
    isolated low-value cells that only appear at low fractions.
    """
    fractions = [0.10, 0.50, 1.00]
    maps = []

    shared_xedges = shared_yedges = None
    for frac in fractions:
        xedges, yedges, mean_map, count_map, meta = build_slice_map(
            dump_path, plane, quantity, times,
            n_bins=n_bins, coord_lim=coord_lim,
            fraction=frac, interpolate=interpolate,
        )
        if shared_xedges is None:
            shared_xedges, shared_yedges = xedges, yedges
        maps.append((mean_map, count_map, meta))

    cm_to_kAU = (1 * u.cm).to(u.au).value / 1e3
    cm3_to_m3 = (1 * u.cm**-3).to(u.m**-3).value

    # ---- Colour limits: use the 100% map as ground truth ----
    # Apply the same unit conversion used when plotting (cm⁻³ → m⁻³ for
    # density) so that vmin/vmax are in the same units as the plotted data.
    mean_map_100, _, _ = maps[-1]
    data_100 = mean_map_100 * cm3_to_m3 if quantity == "density" else mean_map_100
    fp_100 = data_100[np.isfinite(data_100) & (data_100 > 0)]
    if not len(fp_100):
        print("  No finite positive values in 100% map — aborting.")
        return
    _vmin = vmin if vmin is not None else fp_100.min()
    _vmax = vmax if vmax is not None else fp_100.max()
    norm  = (mcolors.LogNorm(vmin=_vmin, vmax=_vmax) if log_scale
             else mcolors.Normalize(vmin=_vmin, vmax=_vmax))

    _, _, meta_ref = maps[-1]
    xedges_kAU = shared_xedges * cm_to_kAU
    yedges_kAU = shared_yedges * cm_to_kAU

    # ---- Figure: reserve space for a dedicated colorbar axis ----
    fig, axes = plt.subplots(
        1, 3,
        figsize=(21, 7),
        dpi=300,
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.905, 0.12, 0.018, 0.74])

    interp_tag = " (interpolated)" if interpolate else ""
    for ax, frac, (mean_map, _, meta) in zip(axes, fractions, maps):
        data = mean_map.T * cm3_to_m3 if quantity == "density" else mean_map.T
        ax.pcolormesh(
            xedges_kAU, yedges_kAU,
            np.ma.masked_invalid(data),
            cmap=cmap, norm=norm, rasterized=True, shading="flat",
        )
        frac_pct = int(frac * 100)
        ax.set_title(rf"{frac_pct}\% of particles{interp_tag}", fontsize=13)
        ax.set_xlabel(meta["h_label"].replace("[cm]", r"[$10^3$ AU]"), fontsize=12)
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.ticklabel_format(style="plain")

    axes[0].set_ylabel(
        meta_ref["v_label"].replace("[cm]", r"[$10^3$ AU]"), fontsize=12
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
    )
    cbar.set_label(meta_ref["q_label"], fontsize=12)

    fig.suptitle(f"Dump {meta_ref['dump_name']}", fontsize=14, y=0.90)

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
            "Plot a 2-D spatial slice (xy or xz) of any scalar field "
            "from a single HDF5 snapshot."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--dump", type=str, default=None,
                     help="Explicit path to a single HDF5 dump file.")
    src.add_argument("--dump-index", type=int, default=None, metavar="N",
                     help="0-based index into the sorted dump_*.h5 list (negative OK).")
    src.add_argument("--list-dumps", action="store_true",
                     help="Print all dump files with indices and sizes, then exit.")

    parser.add_argument("--dump-dir", type=str, default=str(DUMP_DIR))
    parser.add_argument("--plane", choices=["xy", "xz"], default="xy")
    parser.add_argument("--coord-lim", type=float, default=None, metavar="L")
    parser.add_argument(
        "--quantity", type=str, nargs="+", default=["density"],
        metavar="Q",
        help=(
            "One or more fields to plot — loops over each in turn. "
            "E.g. --quantity CO CH4 SiO.  Default: density."
        ),
    )
    parser.add_argument("--times", type=str, default=None, metavar="FIELD2")
    parser.add_argument("--n-bins", type=int, default=N_SLICE_BINS)

    # Subsampling
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of particles to include (0-1, default 1.0).",
    )
    parser.add_argument(
        "--interpolate", action="store_true", default=False,
        help="Fill empty cells by nearest-neighbour interpolation.",
    )
    parser.add_argument(
        "--compare-fractions", action="store_true", default=False,
        help="Plot 10%%, 50%%, 100%% side-by-side in one figure.",
    )

    # Colour scale
    parser.add_argument("--log",    dest="log_scale", action="store_true",  default=True)
    parser.add_argument("--no-log", dest="log_scale", action="store_false")
    parser.add_argument("--cmap",   type=str, default="inferno")
    parser.add_argument("--vmin",   type=float, default=None)
    parser.add_argument("--vmax",   type=float, default=None)

    parser.add_argument("--chemistry",  choices=["Crich", "Orich"], default="Crich")
    parser.add_argument("--n-tasks",    type=int, default=32)
    parser.add_argument("--show",       action="store_true")
    parser.add_argument("--dark-mode",  action="store_true")

    mol_list = ["CO", "CH2", "CH3", "CH4", "HCl", "CH3CN", "SiO", "HCN", "CN", "HC3N", "HC5N", "HC7N", "C2H", "C4H", "C6H", "SiC", "SiN", "H2CS", "H2CO"]
    if is_interactive():
        args = parser.parse_args([
            "--dump-index", "1581",
            "--plane",      "xz",
            "--quantity",   *mol_list,
            # "--quantity",   "CO",
            "--chemistry",  "Crich",
            "--compare-fractions",
            "--interpolate",
            # "--show",
        ])
    else:
        args = parser.parse_args()

    dump_dir = Path(args.dump_dir)

    if args.list_dumps:
        dumps = get_all_dumps(dump_dir)
        if not dumps:
            print(f"No dump_*.h5 files found in {dump_dir}")
            sys.exit(1)
        print(f"\n{len(dumps)} dump files in {dump_dir}:\n")
        for i, p in enumerate(dumps):
            size_mb = p.stat().st_size / 1e6
            print(f"  [{i:4d}]  {p.name}  ({size_mb:.0f} MB)")
        sys.exit(0)

    set_plot_style(args.dark_mode)

    save_dir = SAVE_DIR_BASE / args.chemistry / "slice"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_ext = "pdf" if args.dark_mode else "png"

    dump_path = resolve_dump(dump_dir, args.dump, args.dump_index)
    dump_stem = dump_path.stem.split("_", 1)[1]  # e.g. "01600"

    # ---- Loop over every requested quantity --------------------------------
    for quantity in args.quantity:
        print(f"\n{'='*60}\nQuantity: {quantity}\n{'='*60}")

        if quantity == "density":
            qty_str = f"n{args.times}" if args.times else "density"
        else:
            qty_str = f"{quantity}{args.times}" if args.times else quantity

        if args.compare_fractions:
            interp_tag = "_interp" if args.interpolate else ""
            save_path = save_dir / (
                f"slice_{args.plane}_{qty_str}_{dump_stem}_frac_compare{interp_tag}.{save_ext}"
            )
            plot_fraction_compare(
                dump_path   = dump_path,
                plane       = args.plane,
                quantity    = quantity,
                times       = args.times,
                n_bins      = args.n_bins,
                coord_lim   = args.coord_lim,
                interpolate = args.interpolate,
                log_scale   = args.log_scale,
                cmap        = args.cmap,
                vmin        = args.vmin,
                vmax        = args.vmax,
                save_path   = save_path,
                show        = args.show,
                dark_mode   = args.dark_mode,
            )
        else:
            frac_tag   = f"_f{int(args.fraction * 100):03d}" if args.fraction < 1.0 - 1e-9 else ""
            interp_tag = "_interp" if args.interpolate else ""
            save_path  = save_dir / (
                f"slice_{args.plane}_{qty_str}_{dump_stem}{frac_tag}{interp_tag}.{save_ext}"
            )
            xedges, yedges, mean_map, count_map, meta = build_slice_map(
                dump_path   = dump_path,
                plane       = args.plane,
                quantity    = quantity,
                times       = args.times,
                n_bins      = args.n_bins,
                coord_lim   = args.coord_lim,
                fraction    = args.fraction,
                interpolate = args.interpolate,
            )
            plot_slice_map(
                xedges    = xedges,
                yedges    = yedges,
                mean_map  = mean_map,
                meta      = meta,
                quantity  = quantity,
                log_scale = args.log_scale,
                cmap      = args.cmap,
                vmin      = args.vmin,
                vmax      = args.vmax,
                save_path = save_path,
                show      = args.show,
                dark_mode = args.dark_mode,
            )


if __name__ == "__main__":
    main()