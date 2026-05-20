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
    python plot_slice.py --dump-index 42 --plane xz \\
           --quantity density --times CO

    # Explicit path, custom slice thickness and number of bins
    python plot_slice.py --dump /path/to/dump_0042.h5 \\
           --plane xy --quantity CO --slice-frac 0.05 --n-bins 256

Optional flags
--------------
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

    # Convention 1: separate 1-D arrays
    if all(k in p for k in ("x", "y", "z")):
        return p["x"][:], p["y"][:], p["z"][:]

    # Convention 2: packed (N, 3) array
    for key in ("pos", "xyz", "coordinates", "coords"):
        if key in p:
            arr = p[key][:]                    # shape (N, 3)
            return arr[:, 0], arr[:, 1], arr[:, 2]

    # Fallback: only radial distance available
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

    Raises KeyError with helpful diagnostics if the field is not found.
    """
    p = f["particles"]

    # Physical parameter?
    if name in PHYS_HDF5_KEY:
        key = PHYS_HDF5_KEY[name]
        if key in p:
            return p[key][:]
        raise KeyError(
            f"Physical parameter '{name}' → HDF5 key '{key}' not found in dump.\n"
            f"Available keys: {sorted(p.keys())}"
        )

    # Chemical species
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
    """Return a sorted list of all dump_*.h5 files in *dump_dir*."""
    return sorted(dump_dir.glob("dump_*.h5"))


def resolve_dump(
    dump_dir:   Path,
    dump_path:  str | None,
    dump_index: int | None,
) -> Path:
    """
    Return the Path of the requested dump file.

    Accepts either an explicit path (*dump_path*) or a 0-based index into
    the sorted list of dump_*.h5 files in *dump_dir*.  Negative indices
    are supported (e.g. -1 for the last dump).
    """
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
        return base                        # physical param: no upper bound
    return base & (arr <= 1.0)            # abundance: must be a fraction


def build_slice_map(
    dump_path:  Path,
    plane:      str,           # "xy" or "xz"
    quantity:   str,           # primary field name
    times:      str | None,    # optional second field for product
    n_bins:     int   = N_SLICE_BINS,
    coord_lim:  float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Read *dump_path*, project **all** particles onto a 2-D grid, and return
    the mean quantity per spatial cell.

    All particles are used regardless of their perpendicular coordinate —
    there is no slab selection.  The validity mask mirrors plot_stats.py:
    physical parameters require val > 0; abundances require 0 < val <= 1.

    Parameters
    ----------
    plane     : "xy" projects onto x/y;  "xz" projects onto x/z
    quantity  : scalar field to map (or first factor of a product)
    times     : optional second field; plotted value becomes q1 × q2
    n_bins    : bins per spatial axis in the output map
    coord_lim : manual half-extent [cm] for both axes; auto-detected if None

    Returns
    -------
    xedges    : (n_bins + 1,) bin edges along horizontal axis
    yedges    : (n_bins + 1,) bin edges along vertical axis
    mean_map  : (n_bins, n_bins) mean quantity per cell (NaN where empty)
    count_map : (n_bins, n_bins) particle count per cell
    meta      : dict of labels and statistics for the plot title
    """
    if plane not in ("xy", "xz"):
        raise ValueError(f"plane must be 'xy' or 'xz', got '{plane!r}'")

    print(f"\nReading {dump_path.name} …")
    with h5py.File(dump_path, "r") as f:
        x_all, y_all, z_all = read_coords(f)
        q1 = read_field(f, quantity)
        q2 = read_field(f, times) if times is not None else None

    if quantity == 'density':
        q_label = rf"n{times} [m$^{-3}$]" if times is not None else r"$n$ [m$^{-3}$]"
    elif quantity == 'temperature':
        q_label = rf"$T${times} [K]" if times is not None else r"$T$ [K]"
    elif quantity == 'av':
        q_label = rf"$A_V${times} [mag]" if times is not None else r"$A_V$ [mag]"
    else:
        q_label = rf"{quantity} abundance (wrt H$_{{\mathrm{{nuc}}}}$)"

    # Axis mapping
    if plane == "xy":
        ah, av_arr = x_all, y_all
        lh, lv     = "x [cm]", "y [cm]"
    else:  # "xz"
        ah, av_arr = x_all, z_all
        lh, lv     = "x [cm]", "z [cm]"

    # Per-field validity masks (mirrors plot_stats.py _accumulate_dump)
    mask = _field_mask(q1, quantity)
    if q2 is not None:
        mask &= _field_mask(q2, times)

    ah_v  = ah[mask]
    av_v  = av_arr[mask]
    q_v   = q1[mask] * q2[mask] if q2 is not None else q1[mask]

    n_total = len(x_all)
    n_valid = int(mask.sum())
    print(f"  {n_valid:,} / {n_total:,} particles pass the validity mask")

    if n_valid == 0:
        raise RuntimeError(
            "No valid particles remain after the validity mask.\n"
            "Check the field name and that the dump contains the requested species."
        )

    # Bin edges (symmetric around 0)
    _clim  = coord_lim if coord_lim is not None else (
        max(np.abs(ah_v).max(), np.abs(av_v).max()) * 1.05
    )
    xedges = np.linspace(-_clim, _clim, n_bins + 1)
    yedges = np.linspace(-_clim, _clim, n_bins + 1)

    # 2-D histograms: count and weighted sum
    count_map, _, _ = np.histogram2d(ah_v, av_v, bins=[xedges, yedges])
    sum_map,   _, _ = np.histogram2d(ah_v, av_v, bins=[xedges, yedges],
                                     weights=q_v)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_map = np.where(
            count_map >= MIN_PER_CELL,
            sum_map / count_map,
            np.nan,
        )

    meta = {
        "h_label":   lh,
        "v_label":   lv,
        "q_label":   q_label,
        "dump_name": dump_path.name.split(".")[0].split("_")[1],  # e.g. "dump_0042"
        "n_total":   n_total,
        "n_valid":   n_valid,
        "plane":     plane,
    }
    return xedges, yedges, mean_map, count_map, meta

# ---------------------------------------------------------------------------
# Plot
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
    """
    Render the 2-D slice map with pcolormesh.

    *mean_map* has shape (n_bins_h, n_bins_v) — horizontal axis first.
    pcolormesh(xedges, yedges, data) expects data shaped (n_v, n_h),
    so we transpose.
    """
    cm_to_kAU = (1 * u.cm).to(u.au).value / 1e3   # scalar factor
    xedges_kAU = xedges * cm_to_kAU
    yedges_kAU = yedges * cm_to_kAU

    cm3_to_m3 = (1 * u.cm**-3).to(u.m**-3).value   # 1e6

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    data = mean_map.T * cm3_to_m3 if quantity == "density" else mean_map.T       

    finite_pos = data[np.isfinite(data) & (data > 0)]
    if len(finite_pos) == 0:
        print("  Warning: no finite positive values to plot — figure will be empty.")
        plt.close(fig)
        return
    
    if log_scale:
        norm = mcolors.LogNorm(vmin=finite_pos.min(), vmax=finite_pos.max())  
    else:
        norm = mcolors.Normalize(vmin=data[np.isfinite(data)].min(), vmax=data[np.isfinite(data)].max())

    masked = np.ma.masked_invalid(data)
    
    pcm = ax.pcolormesh(
        xedges_kAU, yedges_kAU, masked,   # ← converted edges
        cmap       = cmap,
        norm       = norm,
        rasterized = True,
        shading    = "flat",
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(meta["q_label"], fontsize=12)
    # cbar.set_label(rf"n$\mathrm{{{molecule}}}$ [m$^{{-3}}$]", fontsize=12)

    h_label = meta["h_label"].replace("[cm]", r"[$10^3$ AU]")
    v_label = meta["v_label"].replace("[cm]", r"[$10^3$ AU]")
    ax.set_xlabel(h_label, fontsize=13)
    ax.set_ylabel(v_label, fontsize=13)

    title = (f"Dump {meta['dump_name']}"

    )
    ax.set_title(title, fontsize=14)
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
            "from a single HDF5 snapshot or from pre-accumulated data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Data source -------------------------------------------------------
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--dump", type=str, default=None,
        help="Explicit path to a single HDF5 dump file.",
    )
    src.add_argument(
        "--dump-index", type=int, default=None,
        metavar="N",
        help="0-based index into the sorted dump_*.h5 list (negative OK).",
    )
    src.add_argument(
        "--list-dumps", action="store_true",
        help="Print all available dump files with indices and sizes, then exit.",
    )

    parser.add_argument(
        "--dump-dir", type=str, default=str(DUMP_DIR),
        help=f"Directory containing dump_*.h5 files (default: {DUMP_DIR}).",
    )

    # ---- Slice geometry ----------------------------------------------------
    parser.add_argument(
        "--plane", choices=["xy", "xz"], default="xy",
        help="Projection plane (default: xy).",
    )
    parser.add_argument(
        "--coord-lim", type=float, default=None,
        metavar="L",
        help="Manual half-extent of the plotted region [cm]. Auto-detected if omitted.",
    )

    # ---- Quantity ----------------------------------------------------------
    parser.add_argument(
        "--quantity", type=str, default="density",
        help=(
            "Field to plot: 'density', 'temperature', 'av', or any species "
            "name (e.g. 'CO', 'HCl', 'SiO').  Default: density."
        ),
    )
    parser.add_argument(
        "--times", type=str, default=None,
        metavar="FIELD2",
        help=(
            "Optional second field; plotted value = --quantity × --times. "
            "Example: --quantity density --times CO  →  n × f_CO."
        ),
    )

    # ---- Grid --------------------------------------------------------------
    parser.add_argument(
        "--n-bins", type=int, default=N_SLICE_BINS,
        help=f"Bins per spatial axis (default: {N_SLICE_BINS}).",
    )

    # ---- Colour scale ------------------------------------------------------
    parser.add_argument(
        "--log", dest="log_scale", action="store_true", default=True,
        help="Logarithmic colour scale (default: on).",
    )
    parser.add_argument(
        "--no-log", dest="log_scale", action="store_false",
        help="Linear colour scale.",
    )
    parser.add_argument("--cmap", type=str, default="inferno",
                        help="Matplotlib colourmap (default: inferno).")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Manual colour-scale lower limit.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Manual colour-scale upper limit.")

    # ---- Output ------------------------------------------------------------
    parser.add_argument(
        "--chemistry", choices=["Crich", "Orich"], default="Crich",
        help="Chemistry type (used for output directory and accumulated mode).",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=32,
        help="Number of accumulate tasks (accumulated mode only).",
    )
    parser.add_argument("--show",      action="store_true",
                        help="Display the figure interactively.")
    parser.add_argument("--dark-mode", action="store_true",
                        help="Use a dark background and adapted colour scheme.")

    # Interactive / notebook defaults
    if is_interactive():
        args = parser.parse_args([
            "--dump-index", "1581",
            "--plane",      "xz",
            "--quantity",   "density",
            "--times",      "CO",
            "--chemistry",  "Crich",
            "--show",
        ])
    else:
        args = parser.parse_args()

    dump_dir = Path(args.dump_dir)

    # ---- List-dumps mode ---------------------------------------------------
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

    save_dir = SAVE_DIR_BASE / args.chemistry / 'slice'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_ext = "pdf" if args.dark_mode else "png"

    # Build a concise filename stem from the quantity
    if args.quantity == "density":
        qty_str = f"n{args.times}" if args.times else "density"
    else:
        qty_str = (
            f"{args.quantity}{args.times}" if args.times else args.quantity
    )
    # ---- Single-dump mode --------------------------------------------------
    dump_path = resolve_dump(dump_dir, args.dump, args.dump_index)

    xedges, yedges, mean_map, count_map, meta = build_slice_map(
        dump_path  = dump_path,
        plane      = args.plane,
        quantity   = args.quantity,
        times      = args.times,
        n_bins     = args.n_bins,
        coord_lim  = args.coord_lim,
    )

    dump_stem = dump_path.stem.split("_")[-1]  # e.g. "0042" from /../dump_0042
    save_path = save_dir / f"slice_{args.plane}_{qty_str}_{dump_stem}.{save_ext}"
    plot_slice_map(
        xedges    = xedges,
        yedges    = yedges,
        mean_map  = mean_map,
        meta      = meta,
        quantity  = args.quantity,
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