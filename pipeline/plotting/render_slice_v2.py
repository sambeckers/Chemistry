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
- optional column-density weighting (abundance × n_H2 [cm^-3])
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from astropy import units as u
import matplotlib

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
    "density":     "density",
    "temperature": "temp",
    "av":          "av",
}

# Colourmap and label defaults per physical parameter used in overview plot
PHYS_OVERVIEW_DEFAULTS = {
    "temperature": dict(cmap="gist_heat",   label=r"$\log_{10}(T)$ [K]"),
    "density":     dict(cmap="gist_heat",   label=r"$\log_{10}(n)$ [cm$^{-3}$]"),
    "av":          dict(cmap="gist_heat",   label=r"$\log_{10}(A_V)$ [mag]"),
}

PARENT_DAUGHTER_PAIRS = [
    ("C2H2", "C2H"),
    ("HCN",  "CN"),
]

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
    """
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr
    if nan_mask.all():
        return arr

    distances = distance_transform_edt(nan_mask)
    close_mask = nan_mask & (distances <= max_distance)

    if close_mask.any():
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

    if smooth_sigma > 0:
        has_data = np.isfinite(filled) & (filled > 0)
        log_arr  = np.where(has_data, np.log10(filled), 0.0)

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
# Chemistry (and physical param) field attachment
# ===========================================================================
def assign_chemistry_fields(
    sdf,
    dump_path: Path,
    fields: list[str],
):
    """Attach chemistry abundance columns (and physical HDF5 fields) to a
    Sarracen DataFrame.

    Parameters
    ----------
    sdf:
        Sarracen particle DataFrame (modified in-place).
    dump_path:
        Path to the HDF5 chemistry dump file.
    fields:
        List of field names to attach.  Both physical and chemistry fields
        are handled.

    Returns
    -------
    sdf
        The same DataFrame with new columns added for each requested field.
    """
    if not fields:
        return sdf

    sdf_ids = sdf["iorig"].to_numpy(dtype=np.int64)

    raw_data = {}

    with h5py.File(dump_path, "r") as f:
        hdf5_ids = f["particles/id"][:].astype(np.int64)
        for name in fields:
            raw_data[name] = read_field_from_hdf5(f, name)

    sort_idx = np.argsort(hdf5_ids)
    hdf5_ids_sorted = hdf5_ids[sort_idx]

    ins = np.searchsorted(hdf5_ids_sorted, sdf_ids)
    ins_clamped = np.clip(ins, 0, len(hdf5_ids_sorted) - 1)
    matched = hdf5_ids_sorted[ins_clamped] == sdf_ids

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
    """Resolve a Phantom binary path and an HDF5 dump path from CLI arguments."""
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
    """Return a LaTeX colourbar label for the given quantity."""
    dw = r" ($\rho$-weighted)" if dens_weight else ""

    if quantity == "density":
        return r"$n$ [cm$^{-2}$]"
    if quantity == "temperature":
        return rf"$T${dw} [K]"
    if quantity == "av":
        return rf"$A_V${dw} [mag]"

    if col_dens:
        spec = format_species_label(quantity)
        spec_inner = spec.strip("$")
        return rf"$n^{{\mathrm{{{spec_inner}}}}}$ [m$^{{-3}}$]"

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

    Physical parameters ('density', 'temperature', 'av') are now read from
    the HDF5 dump and attached to the SDF under the same key name, so
    Sarracen's render() can find them regardless of how read_phantom names
    its internal columns.
    """
    print(f"\nLoading {phantom_path.name}")

    sdf, _ = sarracen.read_phantom(phantom_path)

    # Always read the requested field from HDF5 — this covers both physical
    # params and chemistry species, and avoids column-name mismatches.
    sdf = assign_chemistry_fields(sdf, dump_path, [quantity])

    n_total = len(sdf)

    if fraction < 1.0:
        ids = sdf["iorig"].to_numpy(dtype=np.int64)
        keep = _particle_keep_mask(ids, fraction)
        sdf = sdf[keep]
        print(f"Retained {keep.sum():,} / {n_total:,} particles")

    # ---- Optional: multiply abundance by H2 column density [cm^-3] --------
    if col_dens and quantity not in PHYS_HDF5_KEY:
        with h5py.File(dump_path, "r") as f:
            hdf5_ids = f["particles/id"][:].astype(np.int64)
            dens_raw = f["particles/density"][:]        # [cm^-3]

        sdf_ids         = sdf["iorig"].to_numpy(dtype=np.int64)
        sort_idx        = np.argsort(hdf5_ids)
        hdf5_ids_sorted = hdf5_ids[sort_idx]
        ins             = np.searchsorted(hdf5_ids_sorted, sdf_ids)
        ins_clamped     = np.clip(ins, 0, len(hdf5_ids_sorted) - 1)
        matched         = hdf5_ids_sorted[ins_clamped] == sdf_ids

        dens_matched  = np.where(matched, dens_raw[sort_idx][ins_clamped], np.nan)
        cm3_to_m3 = (1 * u.cm**-3).to(u.m**-3).value
        sdf[quantity] = sdf[quantity] * dens_matched * cm3_to_m3
        print(f"Multiplied {quantity} by H2 column density → units: cm^-3")

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
    """Format AU axis tick values as 10^3 AU."""
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
    """Apply standard axis labels, tick formatting, and title to a render axes."""
    if xlim is not None and xlim > 1000:
        xlabel = r"$x$ [$10^3$ au]"
        ylabel = r"$y$ [$10^3$ au]" if plane == "xy" else r"$z$ [$10^3$ au]"
        ax.xaxis.set_major_formatter(FuncFormatter(kau_tick_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(kau_tick_formatter))
    else:
        xlabel = r"$x$ [au]"
        ylabel = r"$y$ [au]" if plane == "xy" else r"$z$ [au]"

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
    """Fill NaN pixels in the first image on ax using fill_nans_nearest."""
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
    cmap="gist_heat",
    log_scale=True,
    vmin=None,
    vmax=None,
    interpolate=False,
    fraction=1.0,
):
    """Render an SPH quantity onto a Matplotlib axes using Sarracen."""
    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    x_col = "x"
    y_col = "y" if plane == "xy" else "z"

    render_kwargs = dict(
        x=x_col,
        y=y_col,
        cmap=cmap,
        cbar=False,
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
    """Produce and optionally save a single SPH render for one quantity."""
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
    """Return (nrows, ncols) for a panel grid containing *n* axes."""
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
    """Produce a multi-panel figure comparing renders at several particle fractions."""
    if fractions is None:
        fractions = FRACTIONS_3

    n = len(fractions)
    nrows, ncols = _grid_shape(n)

    panel_size = 7
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    all_ax = list(axes.flat)
    for spare in all_ax[n:]:
        spare.set_visible(False)

    cbar_width   = 0.015
    cbar_gap     = 0.001
    right_margin = 0.001
    right = 1.0 - cbar_width - cbar_gap - right_margin
    fig.subplots_adjust(right=right, wspace=-0.1, hspace=0.08)

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

    ref_frac = max(fractions)
    sdf_full, _ = all_sdf[ref_frac]
    vals = sdf_full[quantity].to_numpy()
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vmin is None:
        vmin = vals.min()
    if vmax is None:
        vmax = vals.max()

    for idx, (ax, frac) in enumerate(zip(all_ax[:n], fractions)):
        row_idx = idx // ncols
        col_idx = idx % ncols

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
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        format_render_axes(
            ax, quantity, plane, meta,
            xlim=xlim,
            xsec=xsec,
            interpolate=interpolate,
            q_in_title=False,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_title=False,
        )
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')

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

    cbar_left = right + cbar_gap - 0.02

    for row_idx in range(nrows):
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

        mappable = row_axes[0].images[0]
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label(quantity_label(quantity, dens_weight, col_dens=col_dens), fontsize=FONT_SIZE)
        cbar.ax.tick_params(labelsize=FONT_SIZE)

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
    # ------------------------------------------------------------------
    # Axis labels: x bottom row only, y left column only
    # ------------------------------------------------------------------
    if xlim is not None and xlim > 1000:
        xlab = '$x$ [$10^3$ au]'

        if plane == "xy":
            ylab = '$y$ [$10^3$ au]'
        elif plane == "xz":
            ylab = '$z$ [$10^3$ au]'
        else:
            ylab = None
    else:
        xlab = '$x$ [au]'

        if plane == "xy":
            ylab = '$y$ [au]'
        elif plane == "xz":
            ylab = '$z$ [au]'
        else:
            ylab = None

    for col_idx in range(ncols):
        ax = axes[nrows - 1, col_idx]
        if ax.get_visible():
            ax.set_xlabel(xlab, fontsize=FONT_SIZE)

    if ylab is not None:
        for row_idx in range(nrows):
            ax = axes[row_idx, 0]
            if ax.get_visible():
                ax.set_ylabel(ylab, fontsize=FONT_SIZE)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ===========================================================================
# Physical parameters overview  (2 rows × 3 cols)
# ===========================================================================
def plot_phys_overview(
    phantom_path: Path,
    dump_path: Path,
    xlim: float | None,
    save_path: Path | None,
    show: bool,
    log_scale: bool = True,
):
    quantities = ["temperature", "density", "av"]
    planes     = ["xy", "xz"]

    print(f"\nLoading {phantom_path.name} (phys overview)")
    sdf_raw, _ = sarracen.read_phantom(phantom_path)
    sdf_raw = assign_chemistry_fields(sdf_raw, dump_path, quantities)
    sdf_raw = sdf_raw.dropna(subset=quantities)
    print(f"Rendering with {len(sdf_raw):,} particles")

    meta = {
        "fraction": 1.0,
        "n_particles": len(sdf_raw),
        "dump_name": phantom_path.name.split("_", 1)[1],
    }

    panel_size = 6
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        dpi=300,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # ------------------------------------------------------------------
    # Render all panels
    # ------------------------------------------------------------------
    for col_idx, quantity in enumerate(quantities):
        defaults = PHYS_OVERVIEW_DEFAULTS[quantity]
        cmap = defaults["cmap"]

        vals = sdf_raw[quantity].to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        vmin_q = float(vals.min())
        vmax_q = float(vals.max())

        for row_idx, plane in enumerate(planes):
            ax = axes[row_idx, col_idx]

            render_quantity(
                sdf=sdf_raw,
                quantity=quantity,
                plane=plane,
                ax=ax,
                xlim=xlim,
                xsec=None,
                dens_weight=False,
                cmap=cmap,
                log_scale=log_scale,
                vmin=vmin_q,
                vmax=vmax_q,
                interpolate=False,
                fraction=1.0,
            )

            # Strip everything sarracen/format_render_axes may set
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')

            format_render_axes(
                ax, quantity, plane, meta,
                xlim=xlim,
                xsec=None,
                q_in_title=False,
                show_xlabel=False,
                show_ylabel=False,
                show_title=False,
            )

            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')

            if xlim is not None:
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-xlim, xlim)
            else:
                # Fall back to global particle extent
                ax.set_xlim(sdf_raw['x'].min(), sdf_raw['x'].max())
                ax.set_ylim(sdf_raw['y'].min(), sdf_raw['y'].max())

    # ------------------------------------------------------------------
    # Axis labels: x bottom row only, y left column only
    # ------------------------------------------------------------------
    if xlim is not None and xlim > 1000:
        for col_idx in range(ncols):
            axes[nrows - 1, col_idx].set_xlabel('$x$ [$10^3$ au]', fontsize=FONT_SIZE)
        axes[0, 0].set_ylabel('$y$ [$10^3$ au]', fontsize=FONT_SIZE)
        axes[1, 0].set_ylabel('$z$ [$10^3$ au]', fontsize=FONT_SIZE)
    else:
        for col_idx in range(ncols):
            axes[nrows - 1, col_idx].set_xlabel('$x$ [au]', fontsize=FONT_SIZE)
        axes[0, 0].set_ylabel('$y$ [au]', fontsize=FONT_SIZE)
        axes[1, 0].set_ylabel('$z$ [au]', fontsize=FONT_SIZE)

    # ------------------------------------------------------------------
    # Layout — after rendering and limit-forcing
    # ------------------------------------------------------------------
    fig.subplots_adjust(
        left=0.07, right=0.99,
        bottom=0.06, top=0.78,
        wspace=-0.3, hspace=0.08,
    )

    fig.canvas.draw()

    # ------------------------------------------------------------------
    # Colorbars above each xy panel
    # Tick order from bottom up: bar → ticks+values → label
    # ------------------------------------------------------------------
    cbar_height = 0.022
    cbar_gap    = 0.025

    for col_idx, quantity in enumerate(quantities):
        defaults  = PHYS_OVERVIEW_DEFAULTS[quantity]
        ax_top    = axes[0, col_idx]
        pos       = ax_top.get_position()

        cbar_ax = fig.add_axes([
            pos.x0,
            pos.y1 + cbar_gap,
            pos.width,
            cbar_height,
        ])

        vals = sdf_raw[quantity].to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        vmin_log = float(np.log10(vals.min()))
        vmax_log = float(np.log10(vals.max()))

        norm = matplotlib.colors.Normalize(vmin=vmin_log, vmax=vmax_log)
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=defaults["cmap"]),
            cax=cbar_ax,
            orientation='horizontal',
        )

        # Ticks + values on top, label above those
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        cbar.ax.tick_params(direction='out', length=5,   width=1, colors='k', which='major')
        cbar.ax.tick_params(direction='out', length=2.5, width=1, colors='k', which='minor')
        cbar.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        cbar.ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
        cbar.ax.tick_params(axis='x', which='minor', labeltop=False)
        cbar.ax.tick_params(labelsize=FONT_SIZE * 0.75)

        cbar.ax.set_title(
            defaults["label"],
            fontsize=FONT_SIZE * 0.85,
            pad=8,   # points of padding above the tick labels
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_parent_daughter_overview(
    phantom_path: Path,
    dump_path: Path,
    plane: str = "xy",
    xlim: float | None = None,
    save_path: Path | None = None,
    show: bool = False,
    log_scale: bool = True,
    cmap: str = "gist_heat",
    pairs: list[tuple[str, str]] | None = None,
):
    """Render a 2 × 2 parent-daughter species overview panel.
 
    Layout
    ------
    Each *column* corresponds to one parent-daughter pair.  Row 0 holds the
    parent, row 1 the daughter.  A shared colourbar (spanning the combined
    value range of both species in that column) is drawn above every column,
    matching the style of ``plot_phys_overview``.
 
    Default pairs: C2H2 / C2H  (col 0)  and  HCN / CN  (col 1).
 
    Parameters
    ----------
    pairs:
        List of ``(parent, daughter)`` name tuples.  Defaults to
        ``PARENT_DAUGHTER_PAIRS``.  Extend freely for other chemistries.
    """
    if pairs is None:
        pairs = PARENT_DAUGHTER_PAIRS
 
    n_cols = len(pairs)            # one column per pair
    n_rows = 2                     # row 0 = parent, row 1 = daughter
 
    # Flatten into a single list for batch HDF5 loading
    all_species = [sp for pair in pairs for sp in pair]
 
    print(f"\nLoading {phantom_path.name} (parent-daughter overview)")
    sdf_raw, _ = sarracen.read_phantom(phantom_path)
    sdf_raw = assign_chemistry_fields(sdf_raw, dump_path, all_species)
    # Keep particles that are valid for *at least one* of the requested species
    sdf_raw = sdf_raw.dropna(subset=all_species, how="all")
    print(f"Rendering with {len(sdf_raw):,} particles")
 
    meta = {
        "fraction": 1.0,
        "n_particles": len(sdf_raw),
        "dump_name": phantom_path.name.split("_", 1)[1],
    }
 
    panel_size = 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        dpi=300,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
 
    # ------------------------------------------------------------------
    # Per-column shared vmin / vmax (log-space boundaries derived from the
    # union of parent + daughter finite positive values)
    # ------------------------------------------------------------------
    col_vmin: dict[int, float] = {}
    col_vmax: dict[int, float] = {}
    for col_idx, (parent, daughter) in enumerate(pairs):
        combined: list[np.ndarray] = []
        for sp in (parent, daughter):
            v = sdf_raw[sp].to_numpy()
            v = v[np.isfinite(v) & (v > 0)]
            combined.append(v)
        all_vals = np.concatenate(combined)
        col_vmin[col_idx] = float(all_vals.min())
        col_vmax[col_idx] = float(all_vals.max())
 
    # ------------------------------------------------------------------
    # Render every panel
    # ------------------------------------------------------------------
    for col_idx, (parent, daughter) in enumerate(pairs):
        for row_idx, species in enumerate((parent, daughter)):
            ax = axes[row_idx, col_idx]
 
            render_quantity(
                sdf=sdf_raw,
                quantity=species,
                plane=plane,
                ax=ax,
                xlim=xlim,
                xsec=None,
                dens_weight=False,
                cmap=cmap,
                log_scale=log_scale,
                vmin=col_vmin[col_idx],
                vmax=col_vmax[col_idx],
                interpolate=False,
                fraction=1.0,
            )
 
            # Clear anything sarracen or format_render_axes might add
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
 
            format_render_axes(
                ax, species, plane, meta,
                xlim=xlim,
                xsec=None,
                q_in_title=False,
                show_xlabel=False,
                show_ylabel=False,
                show_title=False,
            )
 
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
 
            if xlim is not None:
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-xlim, xlim)
 
            # Species label — top-right corner, white bold text
            ax.text(
                0.97, 0.97,
                format_species_label(species),
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=FONT_SIZE,
                color="white",
                fontweight="bold",
            )
 
    # ------------------------------------------------------------------
    # Axis labels: x on the bottom row only, y on the left column only
    # ------------------------------------------------------------------
    y_coord = "y" if plane == "xy" else "z"
    if xlim is not None and xlim > 1000:
        xlab = r"$x$ [$10^3$ au]"
        ylab = rf"${y_coord}$ [$10^3$ au]"
    else:
        xlab = r"$x$ [au]"
        ylab = rf"${y_coord}$ [au]"
 
    for col_idx in range(n_cols):
        axes[n_rows - 1, col_idx].set_xlabel(xlab, fontsize=FONT_SIZE)
    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel(ylab, fontsize=FONT_SIZE)
 
    # ------------------------------------------------------------------
    # Layout — identical margins to plot_phys_overview
    # ------------------------------------------------------------------
    fig.subplots_adjust(
        left=0.07, right=0.99,
        bottom=0.06, top=0.78,
        wspace=-0.3, hspace=0.08,
    )
 
    fig.canvas.draw()
 
    # ------------------------------------------------------------------
    # Shared colorbars above each column
    # Colourbar spans the combined range of parent + daughter in that column.
    # Tick positioning: values + ticks on top, label above (same as phys_overview).
    # ------------------------------------------------------------------
    cbar_height = 0.022
    cbar_gap    = 0.025
 
    for col_idx, (parent, daughter) in enumerate(pairs):
        ax_top = axes[0, col_idx]
        pos    = ax_top.get_position()
 
        cbar_ax = fig.add_axes([
            pos.x0,
            pos.y1 + cbar_gap,
            pos.width,
            cbar_height,
        ])
 
        vmin_log = np.log10(col_vmin[col_idx])
        vmax_log = np.log10(col_vmax[col_idx])
 
        norm = matplotlib.colors.Normalize(vmin=vmin_log, vmax=vmax_log)
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="horizontal",
        )
 
        # Ticks & values on top, label further above
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
 
        cbar.ax.tick_params(direction="out", length=5,   width=1, colors="k", which="major")
        cbar.ax.tick_params(direction="out", length=2.5, width=1, colors="k", which="minor")
        cbar.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        cbar.ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
        cbar.ax.tick_params(axis="x", which="minor", labeltop=False)
        cbar.ax.tick_params(labelsize=FONT_SIZE * 0.75)
 
        # Title: "Parent / Daughter  log₁₀(abundance rel. H₂)"
        parent_label   = format_species_label(parent)
        daughter_label = format_species_label(daughter)
        cbar.ax.set_title(
            rf"{parent_label} / {daughter_label}"
            "\n"
            r"$\log_{10}$(abundance rel. H$_2$)",
            fontsize=FONT_SIZE * 0.85,
            pad=8,
        )
 
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

            # # dens_weight
            # dens_weight_list.append(
            #     row["dens_weight"].strip().lower() == "true"
            # )

            # xsec
            xsec = row["xsec"]
            xsec_list.append(
                None if xsec in ("", "None")
                else float(xsec)
            )

            # # log
            # log_list.append(
            #     row["log"].strip().lower() == "true"
            # )

    return (
        plane_list,
        xlim_list,
        # dens_weight_list,
        xsec_list,
        # log_list,
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
            "When combined with --compare-fractions, use the extended 9-panel "
            "grid instead of the default 3-panel set."
        ),
    )
    parser.add_argument("--log", nargs="+", dest="log_scale", default=True)
    parser.add_argument("--no-log", dest="log_scale", action="store_false")
    parser.add_argument("--cmap", type=str, default="gist_heat")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dark-mode", action="store_true")
    parser.add_argument("--chemistry", choices=["Crich", "Orich"], default="Crich")
    parser.add_argument("--col-dens", action="store_true", default=False)
    parser.add_argument(
        "--phys-overview",
        action="store_true",
        help=(
            "Render a 2×3 overview panel of all three physical parameters "
            "(temperature, density, Av) in xy and xz projections.  "
            "Only --xlim is needed alongside --dump / --dump-index."
        ),
    )
    parser.add_argument(
        "--parent-daughter",
        action="store_true",
        help=(
            "Render the 2×2 parent-daughter species overview panel "
            "(C2H2/C2H and HCN/CN by default).  "
            "Accepts --plane and --xlim; ignores --quantity and --compare-fractions."
        ),
    )

    # plane_list, xlim_list, xsec_list = load_render_config_lists("render_configs.txt")

    mol_list = ['CO', 'CH2', 'CH3', 'CH4', 'HCl', 'CH3CN', 'SiO', 'HCN', 
                'CN', 'HC3N', 'HC5N', 'HC7N', 'C2H', 'C4H', 'C6H', 'SiC', 
                'SiN', 'H2CS', 'H2CO', 'N2', 'NH3', 'H2S', 'HCP', 'H2O', 'C2H2', 
                'CS', 'SiC2', 'HF', 'C2H4', 'SiS']
    
    if is_interactive():
        args = parser.parse_args([
            "--dump-index", "1581",
            "--plane", "xy",
            # "--quantity", "CO",
            "--xlim", "100",
            "--parent-daughter",
            # "--dens-weight", "True",
            # "--vmin", "1e-8",
            # # "--vmax", "1e-4",
            # "--xsec", "0",
            # "--log", "False",
            # "--compare-fractions",
            # "--nine-fractions",
            # "--interpolate",
            "--col-dens",
            # "--show",
            # "--dark-mode",
        ])
    else:
        args = parser.parse_args()

    # args.plane = plane_list
    # args.xlim = xlim_list
    # args.xsec = xsec_list

    args.plane       = _as_list(args.plane)
    args.xlim        = _as_list(args.xlim)
    args.xsec        = _as_list(args.xsec)
    set_plot_style(args.dark_mode)

    dump_dir    = Path(args.dump_dir)
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

    save_ext  = "pdf" if args.dark_mode else "png"
    dump_stem = dump_path.stem.split("_", 1)[1]

    def _save_dir(xlim: float | None, compare: bool, overview: bool = False) -> Path:
        spatial_tag = "zoom" if (xlim is not None and xlim <= 100) else "full"
        base = SAVE_DIR_BASE / args.chemistry / "render"
        if overview:
            d = base / "phys_overview"
        elif compare:
            d = base / "compare" / spatial_tag
        else:
            d = base / spatial_tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Physical overview mode — ignores --quantity / --plane / --compare
    # ------------------------------------------------------------------
    if args.phys_overview:
        xlim = args.xlim[0]   # take the first (and typically only) value
        save_dir  = _save_dir(xlim, compare=False, overview=True)
        xlim_tag  = f"_{xlim:.0f}AU" if xlim is not None else ""
        save_path = save_dir / f"phys_{dump_stem}{xlim_tag}.{save_ext}"
        plot_phys_overview(
            phantom_path=phantom_path,
            dump_path=dump_path,
            xlim=xlim,
            save_path=save_path,
            show=args.show,
            log_scale=args.log_scale,
        )
        return

    if args.parent_daughter:
        plane = args.plane[0]
        xlim  = args.xlim[0]
        save_dir  = _save_dir(xlim, compare=False, overview=True)
        xlim_tag  = f"_{xlim:.0f}AU" if xlim is not None else ""
        save_path = save_dir / f"parent_daughter_{dump_stem}{xlim_tag}.{save_ext}"
        plot_parent_daughter_overview(
            phantom_path=phantom_path,
            dump_path=dump_path,
            plane=plane,
            xlim=xlim,
            save_path=save_path,
            show=args.show,
            log_scale=args.log_scale,
            cmap=args.cmap,
        )
        return

    # ------------------------------------------------------------------
    # Normal (single / compare-fractions) mode
    # ------------------------------------------------------------------
    fractions = FRACTIONS_9 if args.nine_fractions else FRACTIONS_3
    cd_tag    = "n" if args.col_dens else ""

    for quantity in args.quantity:
        for plane, xlim, xsec in zip(args.plane, args.xlim, args.xsec):
            xsec_tag  = f"_xsec{xsec:.0f}"  if xsec is not None else ""
            dw_tag    = "_dw"                if args.dens_weight      else ""
            xlim_tag  = f"_{xlim:.0f}AU"    if xlim is not None else ""

            if args.compare_fractions:
                frac_set_tag = "_frac9" if args.nine_fractions else "_frac3"
                save_dir  = _save_dir(xlim, compare=True)
                save_path = save_dir / (
                    f"render_{plane}_{cd_tag}{quantity}_{dump_stem}"
                    f"{xsec_tag}{dw_tag}{xlim_tag}{frac_set_tag}_compare.{save_ext}"
                )
                plot_fraction_compare(
                    phantom_path=phantom_path,
                    dump_path=dump_path,
                    quantity=quantity,
                    plane=plane,
                    xlim=xlim,
                    xsec=xsec,
                    dens_weight=args.dens_weight,
                    interpolate=args.interpolate,
                    log_scale=args.log_scale,
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    save_path=save_path,
                    show=args.show,
                    fractions=fractions,
                    col_dens=args.col_dens,
                )

            else:
                save_dir  = _save_dir(xlim, compare=False)
                save_path = save_dir / (
                    f"render_{plane}_{cd_tag}{quantity}_{dump_stem}"
                    f"{xsec_tag}{dw_tag}{xlim_tag}.{save_ext}"
                )
                plot_single_render(
                    phantom_path=phantom_path,
                    dump_path=dump_path,
                    quantity=quantity,
                    plane=plane,
                    xlim=xlim,
                    xsec=xsec,
                    fraction=args.fraction,
                    dens_weight=args.dens_weight,
                    interpolate=args.interpolate,
                    log_scale=args.log_scale,
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    save_path=save_path,
                    show=args.show,
                    col_dens=args.col_dens,
                )


if __name__ == "__main__":
    main()