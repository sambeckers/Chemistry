#!/usr/bin/env python3
"""
render_slice_v2.py

Native Sarracen SPH rendering with:
- deterministic particle subsampling
- compare-fractions mode
- optional interpolation/smoothing
- plot_slice-style architecture
- interactive notebook defaults
- consistent styling and formatting
- AU → 10^3 AU tick formatting
- optional column-density weighting (abundance × n_H2 [cm^-2])
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sarracen
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_PATH
from code_chem.plotting.plot_utils import set_plot_style, format_species_label

# ===========================================================================
# Paths
# ===========================================================================

DUMP_DIR = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
PHANTOM_DIR = Path("/fred/oz304/beckers/v10a09")
SAVE_DIR_BASE = BASE_PATH / "figures/v10a09_out"

# ---------------------------------------------------------------------------
# Spatial grid defaults
# ---------------------------------------------------------------------------
N_SLICE_BINS  = 512     # bins per spatial axis (single-dump mode)
MIN_PER_CELL  = 1       # cells with fewer particles are shown as NaN

# ---------------------------------------------------------------------------
# Global font size — applied consistently to all labels, ticks, and titles
# ---------------------------------------------------------------------------
FONT_SIZE = 24

# ===========================================================================
# Physical parameter mapping
# ===========================================================================
PHYS_HDF5_KEY = {
    "density": "density",
    "temperature": "temp",
    "av": "av",
}

# Fraction sets -----------------------------------------------------------
FRACTIONS_3 = [0.10, 0.50, 1.00]
FRACTIONS_9 = [0.01, 0.05, 0.10, 0.25, 0.33, 0.50, 0.66, 0.80, 1.00]

# ===========================================================================
# Deterministic subsampling
# ===========================================================================
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

# ===========================================================================
# NaN interpolation
# ===========================================================================
def fill_nans_nearest(
    arr: np.ndarray,
    max_distance: float = 100.0,
    smooth_sigma: float = 0,
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

# ===========================================================================
# Name normalization
# ===========================================================================
def normalise_name(name: str) -> str:
    """Map a species/molecule name to its HDF5 dataset key."""
    c = name.strip().lower()
    c = c.replace("+", "_plus").replace("-", "_minus").replace("/", "_")
    c = c.replace("(", "_").replace(")", "_").replace(".", "_")
    c = re.sub(r"[^a-z0-9_]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

# ===========================================================================
# Validity mask
# ===========================================================================
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

# ===========================================================================
# Read chemistry field
# ===========================================================================
def read_field_from_hdf5(f: h5py.File, name: str) -> np.ndarray:
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

# ===========================================================================
# Chemistry matching
# ===========================================================================
def assign_chemistry_fields(
    sdf,
    dump_path: Path,
    fields: list[str],
):
    """Attach chemistry abundance columns to a Sarracen DataFrame.

    Reads particle IDs and abundance arrays from the HDF5 dump, then
    matches them to the particles already loaded in sdf via their
    iorig IDs. Unmatched particles receive NaN. Columns that fail the
    validity check (_field_mask) are also set to NaN.

    Parameters
    ----------
    sdf:
        Sarracen particle DataFrame (modified in-place).
    dump_path:
        Path to the HDF5 chemistry dump file.
    fields:
        List of field names to attach. Physical fields listed in
        PHYS_HDF5_KEY are skipped silently.

    Returns
    -------
    sdf
        The same DataFrame with new columns added for each chemistry field.
    """
    chem_fields_needed = [f for f in fields if f not in PHYS_HDF5_KEY]

    if not chem_fields_needed:
        return sdf

    sdf_ids = sdf["iorig"].to_numpy(dtype=np.int64)

    raw_data = {}

    with h5py.File(dump_path, "r") as f:
        chem_ids = f["particles/id"][:].astype(np.int64)
        for name in chem_fields_needed:
            raw_data[name] = read_field_from_hdf5(f, name)

    sort_idx = np.argsort(chem_ids)
    chem_ids_sorted = chem_ids[sort_idx]

    ins = np.searchsorted(chem_ids_sorted, sdf_ids)
    ins_clamped = np.clip(ins, 0, len(chem_ids_sorted) - 1)
    matched = chem_ids_sorted[ins_clamped] == sdf_ids

    for name, raw in raw_data.items():
        sorted_vals = raw[sort_idx]
        vals = np.where(matched, sorted_vals[ins_clamped], np.nan)
        mask = _field_mask(vals, name)
        sdf[name] = np.where(mask, vals, np.nan)

    return sdf

# ===========================================================================
# Dump helpers
# ===========================================================================
def get_all_dumps(dump_dir: Path = DUMP_DIR):
    return sorted(dump_dir.glob("dump_*.h5"))

def resolve_dump_pair(
    dump_dir: Path,
    phantom_dir: Path,
    dump_path: str | None,
    dump_index: int | None,
):
    """Resolve a Phantom binary path and an HDF5 dump path from CLI arguments.

    Either dump_path (an explicit file path) or dump_index (a
    zero-based index into the sorted dump list) must be supplied.

    Parameters
    ----------
    dump_dir:
        Directory containing dump_*.h5 files.
    phantom_dir:
        Directory containing Phantom wind_* binary files.
    dump_path:
        Explicit path to an HDF5 dump, or None.
    dump_index:
        Index into the sorted dump list, or None.

    Returns
    -------
    tuple[Path, Path]
        (phantom_path, hdf5_path)

    Raises
    ------
    ValueError
        If neither dump_path nor dump_index is provided.
    """
    if dump_path is not None:
        h5 = Path(dump_path)
        stem = h5.stem.split("_", 1)[1]
        phantom = phantom_dir / f"wind_{stem}"
        return phantom, h5

    dumps = get_all_dumps(dump_dir)

    if dump_index is None:
        raise ValueError("Provide --dump or --dump-index")

    h5 = dumps[dump_index]
    stem = h5.stem.split("_", 1)[1]
    phantom = phantom_dir / f"wind_{stem}"

    return phantom, h5

# ===========================================================================
# Labels
# ===========================================================================
def quantity_label(quantity: str, dens_weight: bool, col_dens: bool = False) -> str:
    """Return a LaTeX colourbar label for the given quantity.

    Parameters
    ----------
    quantity:
        Field name (e.g. "density", "temperature", "CO").
    dens_weight:
        Whether the render is density-weighted, which appends a
        ρ-weighted qualifier to the label.
    col_dens:
        Whether the abundance has been multiplied by the H2 column
        density [cm^-2], producing a column number density label
        of the form n_species [cm^-2].

    Returns
    -------
    str
        LaTeX string suitable for colorbar.set_label.
    """
    dw = r" ($\rho$-weighted)" if dens_weight else ""

    if quantity == "density":
        return r"$n$ [cm$^{-3}$]"
    if quantity == "temperature":
        return rf"$T${dw} [K]"
    if quantity == "av":
        return rf"$A_V${dw} [mag]"

    if col_dens:
        # e.g. "CO" → "$n_\mathrm{CO}$ [cm$^{-2}$]"
        spec = format_species_label(quantity)   # LaTeX species string
        spec_inner = spec.strip("$")            # strip outer $ if present
        return rf"$n_{{\mathrm{{{spec_inner}}}}}$ [cm$^{{-2}}$]"

    return rf"Abundance relative to $\mathrm{{H}}_2${dw}"

# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_render_dataframe(
    phantom_path: Path,
    dump_path: Path,
    quantity: str,
    fraction: float = 1.0,
    col_dens: bool = False,
):
    """Load, subsample, and clean a Sarracen DataFrame ready for rendering.

    Reads the Phantom snapshot, optionally attaches chemistry fields from
    the HDF5 dump, applies deterministic particle subsampling, and drops
    rows with NaN in the target quantity.

    If col_dens is True and the quantity is a chemical species, each
    particle's abundance is multiplied by the H2 column density [cm^-2]
    read from the same HDF5 dump (key: particles/density).  The rendered
    quantity then represents the column number density of the species in
    units of cm^-2.

    Parameters
    ----------
    phantom_path:
        Path to the Phantom binary snapshot (wind_*).
    dump_path:
        Path to the corresponding HDF5 chemistry dump.
    quantity:
        Field to render; chemistry fields are fetched from dump_path.
    fraction:
        Fraction of particles to retain (1.0 = all particles).
    col_dens:
        If True, multiply the abundance by the H2 column density
        [cm^-2] stored under particles/density in the HDF5 dump.
        Ignored for physical quantities (density, temperature, av).

    Returns
    -------
    tuple[SarracenDataFrame, dict]
        The cleaned DataFrame and a metadata dict containing fraction,
        n_particles, and dump_name.
    """
    print(f"\nLoading {phantom_path.name}")

    sdf, _ = sarracen.read_phantom(phantom_path)

    if quantity not in PHYS_HDF5_KEY:
        sdf = assign_chemistry_fields(sdf, dump_path, [quantity])

    n_total = len(sdf)

    if fraction < 1.0:
        ids = sdf["iorig"].to_numpy(dtype=np.int64)
        keep = _particle_keep_mask(ids, fraction)
        sdf = sdf[keep]
        print(f"Retained {keep.sum():,} / {n_total:,} particles")

    # ---- Optional: multiply abundance by H2 column density [cm^-2] --------
    if col_dens and quantity not in PHYS_HDF5_KEY:
        with h5py.File(dump_path, "r") as f:
            hdf5_ids = f["particles/id"][:].astype(np.int64)
            dens_raw = f["particles/density"][:]        # [cm^-2]

        sdf_ids         = sdf["iorig"].to_numpy(dtype=np.int64)
        sort_idx        = np.argsort(hdf5_ids)
        hdf5_ids_sorted = hdf5_ids[sort_idx]
        ins             = np.searchsorted(hdf5_ids_sorted, sdf_ids)
        ins_clamped     = np.clip(ins, 0, len(hdf5_ids_sorted) - 1)
        matched         = hdf5_ids_sorted[ins_clamped] == sdf_ids

        dens_matched  = np.where(matched, dens_raw[sort_idx][ins_clamped], np.nan)
        sdf[quantity] = sdf[quantity] * dens_matched
        print(f"Multiplied {quantity} by H2 column density → units: cm^-2")

    # ------------------------------------------------------------------------

    sdf = sdf.dropna(subset=[quantity])
    print(f"Rendering with {len(sdf):,} particles")

    meta = {
        "fraction": fraction,
        "n_particles": len(sdf),
        "dump_name": phantom_path.name.split("_", 1)[1],
    }

    return sdf, meta

# ===========================================================================
# Tick formatting
# ===========================================================================
def kau_tick_formatter(val, pos):
    """Format AU axis tick values as 10^3 AU.

    Intended for use with matplotlib.ticker.FuncFormatter.

    Parameters
    ----------
    val:
        Tick value in AU.
    pos:
        Tick position (unused; required by the FuncFormatter interface).

    Returns
    -------
    str
        Value divided by 1000, formatted to one decimal place.
    """
    return f"{val * 1e-3:.1f}"

# ===========================================================================
# Axis formatting
# ===========================================================================
def format_render_axes(
    ax,
    quantity,
    plane,
    meta,
    xlim=None,
    xsec=None,
    interpolate=False,
    q_in_title=True,
    show_xlabel=True,
    show_ylabel=True,
    show_title=True,
    plot_single=False,
):
    """Apply standard axis labels, tick formatting, and title to a render axes.

    Parameters
    ----------
    ax:
        Matplotlib Axes object to format.
    quantity:
        Field name (used in the title when q_in_title=True).
    plane:
        Projection plane, either "xy" or "xz".
    meta:
        Metadata dict returned by prepare_render_dataframe.
    xlim:
        Half-width of the plotted region in AU, or None.
    xsec:
        Slice position in AU (included in title when given), or None.
    interpolate:
        Append an "(interpolated)" note to the title when True.
    q_in_title:
        Include the quantity name in the title.
    show_xlabel:
        Draw the x-axis label.  Pass False for inner panels in a grid.
    show_ylabel:
        Draw the y-axis label.  Pass False for inner panels in a grid.
    show_title:
        Draw the axes title.  Pass False when a corner annotation is
        used instead (e.g. compare-fractions mode).
    """
    if xlim > 1000:
        xlabel = r"$x$ [$10^3$ AU]"
        ylabel = r"$y$ [$10^3$ AU]" if plane == "xy" else r"$z$ [$10^3$ AU]"
        ax.xaxis.set_major_formatter(FuncFormatter(kau_tick_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(kau_tick_formatter))
    else:
        xlabel = r"$x$ [AU]"
        ylabel = r"$y$ [AU]" if plane == "xy" else r"$z$ [AU]"

    fs = FONT_SIZE if not plot_single else FONT_SIZE / 2
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=fs)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=fs)

    ax.tick_params(
        axis="both", which="major",
        direction="in", length=8, width=1,
        colors="white", top=True, right=True,
        labelsize=fs,
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="in", length=4, width=1,
        colors="white", top=True, right=True,
    )
    ax.xaxis.set_tick_params(labelcolor="black")
    ax.yaxis.set_tick_params(labelcolor="black")
    ax.minorticks_on()
    ax.set_aspect("equal")
    ax.set_facecolor("k")

    if xlim is not None:
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-xlim, xlim)

    if show_title:
        interp_tag = " (interpolated)" if interpolate else ""
        frac_pct = int(meta["fraction"] * 100)
        if q_in_title:
            title = rf"{format_species_label(quantity)} - {frac_pct}\% particles{interp_tag}"
        else:
            title = rf"{frac_pct}\%{interp_tag}"

        if xsec is not None:
            title += rf", slice at z={xsec:.0f} AU"

        ax.set_title(title, fontsize=fs)

# ===========================================================================
# Post-render interpolation
# ===========================================================================
def interpolate_rendered_image(ax):
    """Fill NaN pixels in the first image on ax using fill_nans_nearest.

    Operates directly on the image array stored inside the Matplotlib
    AxesImage — no new image is created. Does nothing if ax has no
    images.

    Parameters
    ----------
    ax:
        Matplotlib Axes whose first AxesImage will be modified.
    """
    if not ax.images:
        return

    img = ax.images[0]
    arr = np.asarray(img.get_array(), dtype=float)

    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0] = np.nan

    arr = fill_nans_nearest(arr)
    img.set_array(arr)

# ===========================================================================
# Sarracen rendering
# ===========================================================================
def render_quantity(
    sdf,
    quantity,
    plane="xy",
    ax=None,
    xlim=None,
    xsec=None,
    dens_weight=False,
    cmap="inferno",
    log_scale=True,
    vmin=None,
    vmax=None,
    interpolate=False,
    fraction=1.0,
):
    """Render an SPH quantity onto a Matplotlib axes using Sarracen.

    Wraps sdf.render with consistent defaults and optionally runs
    post-render NaN interpolation when a particle fraction < 1 is used.

    Parameters
    ----------
    sdf:
        Sarracen particle DataFrame.
    quantity:
        Name of the column to render.
    plane:
        Projection plane: "xy" or "xz".
    ax:
        Target Axes; a new figure/axes pair is created if None.
    xlim:
        Half-width of the view window in AU, or None for auto.
    xsec:
        Z (or Y) coordinate of the slice plane in AU, or None for a
        column-integrated render.
    dens_weight:
        If True, weight the render by SPH density.
    cmap:
        Matplotlib colourmap name.
    log_scale:
        Use logarithmic colour normalisation when True.
    vmin:
        Lower bound of the colour scale, or None for auto.
    vmax:
        Upper bound of the colour scale, or None for auto.
    interpolate:
        Fill NaN pixels after rendering (only applied when fraction < 1).
    fraction:
        Particle fraction used for this render (passed through for the
        interpolation guard).

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the render was drawn.
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    x_col = "x"
    y_col = "y" if plane == "xy" else "z"

    render_kwargs = dict(
        x=x_col,
        y=y_col,
        cmap=cmap,
        cbar=False,
        # vmin=vmin,
        # vmax=vmax,
        log_scale=log_scale,
        dens_weight=dens_weight,
        ax=ax,
    )

    if xlim is not None:
        render_kwargs["xlim"] = (-xlim, xlim)
        render_kwargs["ylim"] = (-xlim, xlim)

    if xsec is not None:
        render_kwargs["xsec"] = xsec

    sdf.render(quantity, **render_kwargs)

    if interpolate and fraction < 1.0:
        interpolate_rendered_image(ax)

    return ax

# ===========================================================================
# Single render
# ===========================================================================
def plot_single_render(
    phantom_path,
    dump_path,
    quantity,
    plane,
    xlim,
    xsec,
    fraction,
    dens_weight,
    interpolate,
    log_scale,
    cmap,
    vmin,
    vmax,
    save_path,
    show,
    col_dens=False,
):
    """Produce and optionally save a single SPH render for one quantity.

    Combines prepare_render_dataframe, render_quantity,
    format_render_axes, and a colourbar into one complete figure.

    Parameters
    ----------
    phantom_path:
        Path to the Phantom binary snapshot.
    dump_path:
        Path to the HDF5 chemistry dump.
    quantity:
        Field to render.
    plane:
        Projection plane ("xy" or "xz").
    xlim:
        Half-width of the view in AU, or None.
    xsec:
        Slice position in AU, or None for column integration.
    fraction:
        Fraction of particles to use.
    dens_weight:
        Whether to density-weight the render.
    interpolate:
        Whether to fill NaN pixels after rendering.
    log_scale:
        Use logarithmic colour scale.
    cmap:
        Matplotlib colourmap name.
    vmin:
        Minimum colour scale value.
    vmax:
        Maximum colour scale value.
    save_path:
        Output file path, or None to skip saving.
    show:
        If True, display the figure interactively; otherwise close it.
    col_dens:
        If True, multiply abundance by H2 column density [cm^-2] before
        rendering.  Colorbar label changes to n_species [cm^-2].
    """
    sdf, meta = prepare_render_dataframe(
        phantom_path,
        dump_path,
        quantity,
        fraction=fraction,
        col_dens=col_dens,
    )

    fig, ax = plt.subplots(dpi=300)

    render_quantity(
        sdf=sdf,
        quantity=quantity,
        plane=plane,
        ax=ax,
        xlim=xlim,
        xsec=xsec,
        dens_weight=dens_weight,
        cmap=cmap,
        log_scale=log_scale,
        vmin=vmin,
        vmax=vmax,
        interpolate=interpolate,
        fraction=fraction,
    )

    format_render_axes(ax, quantity, plane, meta, xlim=xlim, xsec=xsec, plot_single=True)

    mappable = ax.images[0]
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(quantity_label(quantity, dens_weight, col_dens=col_dens), fontsize=FONT_SIZE / 2)
    cbar.ax.tick_params(labelsize=FONT_SIZE / 2)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ===========================================================================
# Grid shape helper
# ===========================================================================
def _grid_shape(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for a panel grid containing *n* axes.

    Rules
    -----
    * n <= 4  → single row
    * n <= 8  → 2 rows × 4 cols
    * n <= 9  → 3 × 3
    * otherwise → ncols = 4, nrows rounded up
    """
    if n <= 4:
        return 1, n
    if n <= 8:
        return 2, 4
    if n <= 9:
        return 3, 3
    ncols = 4
    return math.ceil(n / ncols), ncols

# ===========================================================================
# Compare fractions
# ===========================================================================
def plot_fraction_compare(
    phantom_path,
    dump_path,
    quantity,
    plane,
    xlim,
    xsec,
    dens_weight,
    interpolate,
    log_scale,
    cmap,
    vmin,
    vmax,
    save_path,
    show,
    fractions=None,
    col_dens=False,
):
    """Produce a multi-panel figure comparing renders at several particle fractions.

    By default three panels are produced (10 %, 50 %, 100 %).  Pass a custom
    list via *fractions* to change both the number of panels and the sampled
    fractions (e.g. FRACTIONS_9 for a 3 × 3 grid).

    Layout details
    --------------
    * A shared colour scale is derived from the full (100 %) particle set so
      that all panels are directly comparable.
    * One colourbar is placed to the right of each row.
    * Axis labels are shown only on outer panels (bottom row for x, left
      column for y).
    * The particle fraction is annotated in the top-right corner of each panel
      instead of appearing in the title.

    Parameters
    ----------
    phantom_path:
        Path to the Phantom binary snapshot.
    dump_path:
        Path to the HDF5 chemistry dump.
    quantity:
        Field to render.
    plane:
        Projection plane ("xy" or "xz").
    xlim:
        Half-width of the view in AU, or None.
    xsec:
        Slice position in AU, or None for column integration.
    dens_weight:
        Whether to density-weight the render.
    interpolate:
        Whether to fill NaN pixels after rendering.
    log_scale:
        Use logarithmic colour scale.
    cmap:
        Matplotlib colourmap name.
    vmin:
        Minimum colour scale value (derived from full data if None).
    vmax:
        Maximum colour scale value (derived from full data if None).
    save_path:
        Output file path, or None to skip saving.
    show:
        If True, display the figure interactively; otherwise close it.
    fractions:
        Ordered list of particle fractions to render.  Defaults to
        FRACTIONS_3 ([0.10, 0.50, 1.00]).  Pass FRACTIONS_9
        for the 3 × 3 grid, or any custom list.
    col_dens:
        If True, multiply abundance by H2 column density [cm^-2] before
        rendering.  Colorbar label changes to n_species [cm^-2].
    """
    if fractions is None:
        fractions = FRACTIONS_3

    n = len(fractions)
    nrows, ncols = _grid_shape(n)

    panel_size = 7          # inches per panel
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,      # always 2-D array, regardless of nrows/ncols
    )

    # Hide spare axes that arise when ncols * nrows > n
    all_ax = list(axes.flat)
    for spare in all_ax[n:]:
        spare.set_visible(False)

    # Reserve space on the right for one colorbar per row.
    # tight_layout must NOT be called afterwards (it would override these).
    cbar_width   = 0.015         # colourbar width in figure fraction
    cbar_gap     = 0.01          # gap between rightmost panel and colourbar
    right_margin = 0.01          # gap to the right of colourbar
    right = 1.0 - cbar_width - cbar_gap - right_margin
    fig.subplots_adjust(right=right, wspace=0.05, hspace=0.08)

    # ------------------------------------------------------------------
    # Load all fractions
    # ------------------------------------------------------------------
    all_sdf = {}
    for frac in fractions:
        sdf, meta = prepare_render_dataframe(
            phantom_path,
            dump_path,
            quantity,
            fraction=frac,
            col_dens=col_dens,
        )
        all_sdf[frac] = (sdf, meta)

    # Derive shared colour scale from the highest fraction in the list.
    ref_frac = max(fractions)
    sdf_full, _ = all_sdf[ref_frac]
    vals = sdf_full[quantity].to_numpy()
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vmin is None:
        vmin = vals.min()
    if vmax is None:
        vmax = vals.max()

    # ------------------------------------------------------------------
    # Render each panel
    # ------------------------------------------------------------------
    for idx, (ax, frac) in enumerate(zip(all_ax[:n], fractions)):
        row_idx = idx // ncols
        col_idx = idx % ncols

        # Only label the bottom row (x) and left column (y)
        show_xlabel = (row_idx == nrows - 1)
        show_ylabel = (col_idx == 0)

        sdf, meta = all_sdf[frac]

        render_quantity(
            sdf=sdf,
            quantity=quantity,
            plane=plane,
            ax=ax,
            xlim=xlim,
            xsec=xsec,
            dens_weight=dens_weight,
            cmap=cmap,
            log_scale=log_scale,
            vmin=vmin,
            vmax=vmax,
            interpolate=interpolate,
            fraction=frac,
        )

        format_render_axes(
            ax, quantity, plane, meta,
            xlim=xlim,
            xsec=xsec,
            interpolate=interpolate,
            q_in_title=False,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_title=False,          # fraction shown as corner annotation
        )

        # Fraction annotation in the top-right corner
        frac_pct = int(frac * 100)
        frac_label = rf"{frac_pct}\%" if frac_pct >= 1 else rf"{frac * 100:.1f}\%"
        ax.text(
            0.97, 0.97,
            frac_label,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=FONT_SIZE,
            color="white",
            fontweight="bold",
        )

    # ------------------------------------------------------------------
    # One colourbar per row, placed to the right of the grid.
    # We read the axes positions *after* subplots_adjust so the bounding
    # boxes reflect the final layout.
    # ------------------------------------------------------------------
    cbar_left = right + cbar_gap   # left edge of all colourbars

    for row_idx in range(nrows):
        # Collect visible axes in this row
        row_axes = [
            axes[row_idx, col_idx]
            for col_idx in range(ncols)
            if axes[row_idx, col_idx].get_visible()
        ]
        if not row_axes:
            continue

        y0 = min(ax.get_position().y0 for ax in row_axes)
        y1 = max(ax.get_position().y1 for ax in row_axes)

        cbar_ax = fig.add_axes([cbar_left, y0, cbar_width, y1 - y0])

        # Use the first rendered panel of this row as the mappable source
        mappable = row_axes[0].images[0]
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label(quantity_label(quantity, dens_weight, col_dens=col_dens), fontsize=FONT_SIZE)
        cbar.ax.tick_params(labelsize=FONT_SIZE)

    # Overall title showing the quantity name
    visible_axes = [ax for ax in all_ax[:n] if ax.get_visible()]

    left_edge  = min(ax.get_position().x0 for ax in visible_axes)
    right_edge = max(ax.get_position().x1 for ax in visible_axes)

    x_center = 0.5 * (left_edge + right_edge)

    y_suptitle = max(ax.get_position().y1 for ax in visible_axes) + 0.02

    y_suptitle = min(y_suptitle, 0.98) if fractions == FRACTIONS_9 else 0.95
    fig.suptitle(
        format_species_label(quantity),
        fontsize=FONT_SIZE,
        fontweight="bold",
        x=x_center,
        y=y_suptitle,
    )

    # Do NOT call tight_layout — it would override subplots_adjust and
    # move the colourbar axes back inside the rightmost panel.

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ===========================================================================
# Render config reader
# ===========================================================================
import csv


def load_render_config_lists(path):
    """
    Read render configuration CSV/TXT file and return Python lists
    matching the argparse variables.

    Expected columns:
        plane,xlim,dens_weight,xsec,log
    """

    plane_list = []
    xlim_list = []
    dens_weight_list = []
    xsec_list = []
    log_list = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            # plane
            plane_list.append(row["plane"])

            # xlim
            xlim = row["xlim"]
            xlim_list.append(
                None if xlim in ("", "None")
                else float(xlim)
            )

            # dens_weight
            dens_weight_list.append(
                row["dens_weight"].strip().lower() == "true"
            )

            # xsec
            xsec = row["xsec"]
            xsec_list.append(
                None if xsec in ("", "None")
                else float(xsec)
            )

            # log
            log_list.append(
                row["log"].strip().lower() == "true"
            )

    return (
        plane_list,
        xlim_list,
        dens_weight_list,
        xsec_list,
        log_list,
    )

def _as_list(val, *, boolean=False):
    """Wrap scalars/None into a single-element list; lists pass through."""
    if isinstance(val, list):
        if boolean:
            return [v if isinstance(v, bool) else v.strip().lower() != "false"
                    for v in val]
        return val
    if val is None:
        return [None]
    if boolean:
        return [val if isinstance(val, bool) else val.strip().lower() != "false"]
    return [val]

# ===========================================================================
# Interactive detection
# ===========================================================================
def is_interactive():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

# ===========================================================================
# Main
# ===========================================================================
def main():
    """Parse command-line arguments and dispatch to the appropriate render function."""
    parser = argparse.ArgumentParser(
        description="Native Sarracen SPH rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--dump", type=str)
    src.add_argument("--dump-index", type=int)
    src.add_argument("--list-dumps", action="store_true")

    parser.add_argument("--dump-dir", type=str, default=str(DUMP_DIR))
    parser.add_argument("--phantom-dir", type=str, default=str(PHANTOM_DIR))
    parser.add_argument("--plane", choices=["xy", "xz"], nargs="+", default="xy")
    parser.add_argument("--quantity", nargs="+", default=["density"])
    parser.add_argument("--xlim", nargs="+", type=float, default=None)
    parser.add_argument("--xsec", nargs="+", type=float, default=None)
    parser.add_argument("--dens-weight", nargs="+", default=False)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--compare-fractions", action="store_true")
    parser.add_argument(
        "--nine-fractions",
        action="store_true",
        help=(
            "When combined with --compare-fractions, use the extended 8-panel "
            "grid [1%%, 5%%, 10%%, 33%%, 50%%, 66%%, 80%%, 100%%] instead of "
            "the default 3-panel set [10%%, 50%%, 100%%]."
        ),
    )
    parser.add_argument("--log", nargs="+", dest="log_scale", default=True)
    parser.add_argument("--no-log", dest="log_scale", action="store_false")
    parser.add_argument("--cmap", type=str, default="inferno")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dark-mode", action="store_true")
    parser.add_argument("--chemistry", choices=["Crich", "Orich"], default="Crich")
    parser.add_argument(
        "--col-dens",
        action="store_true",
        default=False,
        help=(
            "Multiply the rendered abundance by the H2 column density "
            "[cm^-2] (key: particles/density in the HDF5 dump). "
            "The colorbar label changes to n_species [cm^-2] and the "
            "output filename gains a '_cd' suffix."
        ),
    )

    plane_list, xlim_list, dens_weight_list, xsec_list, log_list = load_render_config_lists("render_configs.txt")
    mol_list = ["CO", "CH2", "CH3", "CH4", "HCl", "CH3CN", "SiO", "HCN", "CN", "HC3N", "HC5N", "HC7N", "C2H", "C4H", "C6H", "SiC", "SiN", "H2CS", "H2CO"]

    if is_interactive():
        args = parser.parse_args([
            "--dump-index", "1581",
            "--plane", "xy",
            "--quantity", "CO",
            "--xlim", "100",
            # "--dens-weight", "True",
            # "--vmin", "1e-8",
            # "--vmax", "1e-4",
            "--xsec", "0",
            "--log", "False",
            # "--compare-fractions",
            # "--nine-fractions",
            # "--interpolate",
            "--col-dens",
            "--show",
            # "--dark-mode",
        ])
    else:
        args = parser.parse_args()

    args.plane       = _as_list(args.plane)
    args.xlim        = _as_list(args.xlim)
    args.dens_weight = _as_list(args.dens_weight, boolean=True)
    args.xsec        = _as_list(args.xsec)
    args.log_scale   = _as_list(args.log_scale, boolean=True)
    set_plot_style(args.dark_mode)

    dump_dir = Path(args.dump_dir)
    phantom_dir = Path(args.phantom_dir)

    if args.list_dumps:
        dumps = get_all_dumps(dump_dir)
        for i, p in enumerate(dumps):
            print(f"[{i:4d}] {p.name}")
        sys.exit(0)

    phantom_path, dump_path = resolve_dump_pair(
        dump_dir,
        phantom_dir,
        args.dump,
        args.dump_index,
    )

    save_dir = SAVE_DIR_BASE / args.chemistry / "render"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_ext = "pdf" if args.dark_mode else "png"
    dump_stem = dump_path.stem.split("_", 1)[1]

    # Resolve which fraction set to use for compare-fractions mode.
    fractions = FRACTIONS_9 if args.nine_fractions else FRACTIONS_3

    # Tag for column-density mode
    cd_tag = "_cd" if args.col_dens else ""

    for quantity in args.quantity:
        for plane, xlim, dens_weight, xsec, log in zip(
            args.plane, args.xlim, args.dens_weight, args.xsec, args.log_scale
        ):
            xsec_tag  = f"_xsec{xsec:.0f}"      if xsec is not None else ""
            dw_tag    = "_dw"                    if dens_weight      else ""
            xlim_tag  = f"_xlim{xlim:.0f}AU"    if xlim is not None else ""

            if args.compare_fractions:
                frac_set_tag = "_frac9" if args.nine_fractions else "_frac3"
                save_path = save_dir / (
                    f"render_{plane}_{quantity}_{dump_stem}"
                    f"{xsec_tag}{dw_tag}{cd_tag}{xlim_tag}{frac_set_tag}_compare.{save_ext}"
                )
                plot_fraction_compare(
                    phantom_path=phantom_path,
                    dump_path=dump_path,
                    quantity=quantity,
                    plane=plane,
                    xlim=xlim,
                    xsec=xsec,
                    dens_weight=dens_weight,
                    interpolate=args.interpolate,
                    log_scale=log,
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    save_path=save_path,
                    show=args.show,
                    fractions=fractions,
                    col_dens=args.col_dens,
                )

            else:
                save_path = save_dir / (
                    f"render_{plane}_{quantity}_{dump_stem}"
                    f"{xsec_tag}{dw_tag}{cd_tag}{xlim_tag}.{save_ext}"
                )

                plot_single_render(
                    phantom_path=phantom_path,
                    dump_path=dump_path,
                    quantity=quantity,
                    plane=plane,
                    xlim=xlim,
                    xsec=xsec,
                    fraction=args.fraction,
                    dens_weight=dens_weight,
                    interpolate=args.interpolate,
                    log_scale=log,
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    save_path=save_path,
                    show=args.show,
                    col_dens=args.col_dens,
                )


if __name__ == "__main__":
    main()