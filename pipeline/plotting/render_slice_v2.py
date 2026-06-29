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
from scipy.spatial import cKDTree
import matplotlib
from itertools import product
import matplotlib.patheffects as pe

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sarracen
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

import importlib.util

BASE_PATH = Path("/fred/oz304/beckers/Chemistry")

# Load plot_utils
_spec2 = importlib.util.spec_from_file_location(
    "plot_utils", BASE_PATH / "code_chem" / "plotting" / "plot_utils.py"
)

_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
set_plot_style = _mod2.set_plot_style
format_species_label = _mod2.format_species_label

# ===========================================================================
# Paths
# ===========================================================================

# DUMP_DIR = Path('/aphid/scratch-3month/sbeckers/v20a25_out/output/dumps')
# PHANTOM_DIR = Path("/fred/oz304/beckers/v20a25")
# SAVE_DIR_BASE = BASE_PATH / "figures/v20a25_out"

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
    if name in PHYS_HDF5_KEY or name == "velocity":
        return base
    return base & (arr <= 1.0)

# ===========================================================================
# Read chemistry field
# ===========================================================================
def read_field_from_hdf5(f: h5py.File, name: str) -> np.ndarray:
    """
    Read a scalar field from the 'particles' group of an open HDF5 file.

    Accepts physical parameter shortcuts ('density', 'temperature', 'av')
    and chemical species names ('CO', 'HCl', 'SiO', ...).
    """
    p = f["particles"]

    if name in PHYS_HDF5_KEY:
        key = PHYS_HDF5_KEY[name]
        if key in p:
            return p[key][:]
        raise KeyError(
            f"Physical parameter '{name}' -> HDF5 key '{key}' not found in dump.\n"
            f"Available keys: {sorted(p.keys())}"
        )

    key = normalise_name(name)
    if key in p:
        return p[key][:]
    raise KeyError(
        f"Species '{name}' -> normalised key '{key}' not found in dump.\n"
        f"Available keys: {sorted(p.keys())}"
    )

def _read_time_yr(dump_path: Path) -> float | None:
    """Read simulation time in years from the HDF5 chemistry dump (stored in seconds)."""
    try:
        with h5py.File(dump_path, "r") as f:
            for loc in ("time", "header/time", "particles/time"):
                try:
                    t_s = float(np.asarray(f[loc]).flat[0])
                    return t_s / (365.25 * 24 * 3600)
                except KeyError:
                    continue
        return None
    except Exception:
        return None


def _add_time_label(fig, time_yr: float, fontsize: int = 10,
                    x: float = 0.05, y: float = 0.98) -> None:
    """Add a figure-level time stamp for animation frames."""
    color = matplotlib.rcParams.get("text.color", "black")
    fig.text(
        x, y,
        rf"$t = {round(time_yr)}$ yr",
        ha="left", va="top",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        transform=fig.transFigure,
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

def assign_velocity_field(sdf):
    """
    Attach a 'velocity' column (speed magnitude, km/s) to a Sarracen
    DataFrame, computed from its own vx/vy/vz columns and unit_velocity
    """
    unit_dist     = sdf._params["udist"]
    unit_time     = sdf._params["utime"]
    unit_velocity = unit_dist / unit_time  # cm/s per code unit

    speed_cms = np.sqrt(sdf["vx"]**2 + sdf["vy"]**2 + sdf["vz"]**2).to_numpy() * unit_velocity
    cms_to_kms = (1 * u.cm / u.s).to(u.km / u.s).value
    speed_kms = speed_cms * cms_to_kms

    mask = _field_mask(speed_kms, "velocity")
    sdf["velocity"] = np.where(mask, speed_kms, np.nan)
    print("Assigned 'velocity' column (km/s) to Sarracen DataFrame from vx/vy/vz")
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
    # dw = r" ($\rho$-weighted)" if dens_weight else ""

    if quantity == "density":
        return r"$n$ [cm$^{-2}$]"
    if quantity == "temperature":
        return rf"$T$ [K]"
    if quantity == "av":
        return rf"$A_V$ [mag]"
    if quantity == "velocity":
        return r"$v$ [km s$^{-1}$]"

    if col_dens:
        spec = format_species_label(quantity)
        spec_inner = spec.strip("$")
        return rf"$n^{{\mathrm{{{spec_inner}}}}}$ [m$^{{-3}}$]"

    return rf"Abundance relative to $\mathrm{{H}}_2$"

# ===========================================================================
# Data preparation
# ===========================================================================
def remove_abundance_outliers(
    sdf,
    quantity: str,
    k: int = 16,
    log_thresh: float = 2.0,   # decades above local median -> outlier
) -> sdf:
    """Remove particles with abundances that are anomalously
    high compared to their local neighbourhood."""
    coords = sdf[["x", "y", "z"]].to_numpy()
    vals   = sdf[quantity].to_numpy(dtype=float)

    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=k + 1)   # col 0 is self

    local_median = np.median(vals[idxs[:, 1:]], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(vals / local_median)

    outlier = np.isfinite(log_ratio) & (log_ratio > log_thresh)
    n_removed = outlier.sum()
    print(f"Abundance outlier filter: removed {n_removed:,} particles "
          f"(>{log_thresh} dex above local median of {k} neighbours)")
    return sdf[~outlier]

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

    sdf, sinks = sarracen.read_phantom(phantom_path)

    # Always read the requested field from HDF5 -- this covers both physical
    # params and chemistry species, and avoids column-name mismatches.
    if quantity == "velocity":
        sdf = assign_velocity_field(sdf)
    else:
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
        print(f"Multiplied {quantity} by H2 column density -> units: cm^-3")

    sdf = sdf.dropna(subset=[quantity])
    sdf = remove_abundance_outliers(sdf, quantity, k=16, log_thresh=2.0)
    print(f"Rendering with {len(sdf):,} particles")

    meta = {
        "fraction": fraction,
        "n_particles": len(sdf),
        "dump_name": phantom_path.name.split("_", 1)[1],
        "sinks": sinks,
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
    small_ticks=None,
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

    tick_size = 4 if small_ticks else 8
    ax.tick_params(
        axis="both", which="major",
        direction="in", length=tick_size, width=1,
        colors="white", top=True, right=True,
        labelsize=fs,
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="in", length=tick_size/2, width=1,
        colors="white", top=True, right=True,
    )
    label_color = plt.rcParams.get("text.color", "black")
    ax.xaxis.set_tick_params(labelcolor=label_color)
    ax.yaxis.set_tick_params(labelcolor=label_color)
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
            title = rf"{format_species_label(quantity)}"
        else:
            title = rf"{frac_pct}\%{interp_tag}"

        if xsec is not None:
            title += rf", cross section at $z$={xsec:.0f} AU"

        if not quantity == "velocity":
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
# Contour radii
# ===========================================================================
def extract_contour_radii(contour_set):
    results = {}
    for i, level in enumerate(contour_set.levels):
        segs = contour_set.allsegs[i]
        if not segs:
            results[level] = np.nan
            continue
        all_points = np.concatenate(segs, axis=0)
        r = np.sqrt(all_points[:, 0]**2 + all_points[:, 1]**2)
        results[level] = np.mean(r)
    return results

def draw_radii_arrows(ax, radii, plane="xy"):
    """Draw arrows from the origin to each contour radius for visual verification."""
    colors = ["white"] * len(radii)  # one per level
    x0, x1, y0, y1 = ax.images[0].get_extent()

    angles = [45, 135, 225, 315]
    for (level, r), color, angle in zip(radii.items(), colors, angles):
        if np.isnan(r):
            continue
        angle = np.radians(angle)
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            color=color,
            fontsize=8,
            ha="center",
        )

    return ax

CONTOUR_COLS = ["molecule", "R_xy_001", "R_xy_050", "R_xz_001", "R_xz_050"]

def save_contour_radii(
    save_path: Path,
    quantity: str,
    plane: str,
    radii: dict,
    contour_values: list,
):
    """
    Append or update contour radii in a tab-separated .txt file.

    Columns: molecule  R_xy_001  R_xy_050  R_xz_001  R_xz_050

    - If the file doesn't exist, it is created with a header.
    - If the molecule row doesn't exist yet, a new row is appended.
    - If the molecule row exists (e.g. xy already written, now writing xz),
      only the relevant columns are updated.
    """
    import csv

    levels = sorted(radii.keys())
    if plane == "xy":
        col_map = {levels[0]: "R_xy_001", levels[-1]: "R_xy_050"}
        fill_cols = {"R_xz_001": "--", "R_xz_050": "--"}
    else:
        col_map = {levels[0]: "R_xz_001", levels[-1]: "R_xz_050"}
        fill_cols = {"R_xy_001": "--", "R_xy_050": "--"}

    new_vals = {}
    for level, col in col_map.items():
        r = radii.get(level, np.nan)
        new_vals[col] = f"{r:.1f}" if np.isfinite(r) else "--"

    save_path = Path(save_path)

    rows = []
    if save_path.exists():
        with open(save_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

    mol_row = next((r for r in rows if r["molecule"] == quantity), None)

    if mol_row is None:
        mol_row = {"molecule": quantity, **fill_cols, **new_vals}
        rows.append(mol_row)
    else:
        mol_row.update(new_vals)

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONTOUR_COLS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Contour radii saved -> {save_path}")

# ===========================================================================
# AGB velocity and wind angle line
# ===========================================================================

def compute_velAGB(sdf, sinks):
    """AGB (sink) velocity vector in cm/s."""
    unit_dist     = sdf._params["udist"]
    unit_time     = sdf._params["utime"]
    unit_velocity = unit_dist / unit_time

    vxAGB = sinks["vx"][0] * unit_velocity
    vyAGB = sinks["vy"][0] * unit_velocity
    vzAGB = sinks["vz"][0] * unit_velocity
    return np.array([vxAGB, vyAGB, vzAGB])

def compute_wind_cone_angle(sinks, sdf, v_wind_kms: float = 10.0) -> float:
    """
    theta = arctan(v_wind / v_orb), where v_orb is the AGB's orbital
    speed in the xy-plane (assumed orbital plane) and v_wind is the
    wind expansion speed [km/s]. This is the canonical half-opening
    angle of the companion-wind interaction cone, measured from the
    orbital (z=0) plane -- i.e. from the horizontal line in an xz slice.
    """
    unit_dist     = sdf._params["udist"]
    unit_time     = sdf._params["utime"]
    unit_velocity = unit_dist / unit_time
    cms_to_kms    = (1 * u.cm / u.s).to(u.km / u.s).value

    vxAGB = sinks["vx"][0] * unit_velocity * cms_to_kms
    vzAGB = sinks["vz"][0] * unit_velocity * cms_to_kms
    v_orb = np.hypot(vxAGB, vzAGB)   # km/s
    print(v_orb)

    return np.arctan2(v_wind_kms, v_orb)

import matplotlib.patheffects as mpl_pe
def draw_wind_cone_lines(ax, theta, xlim=None, color="k", lw=1, ls="-"):
    """
    Draw an X through the origin at +/-theta from the horizontal
    (z=0) line -- the wind-cone opening angle, with an arc annotation
    showing the narrow opening angle between the lines on the left side.
    """
    if xlim is not None:
        r = xlim
    else:
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        r = max(abs(v) for v in (*xlims, *ylims)) + 100

    dx = -r * np.sin(theta)
    dz =  r * np.cos(theta)

    ax.plot([-dx, dx], [-dz, dz], color=color, lw=lw, ls=ls, alpha=0.8)
    ax.plot([-dx, dx], [dz, -dz], color=color, lw=lw, ls=ls, alpha=0.8)

    # Narrow angle in the left-side notch between the two arms
    narrow_deg = 2.0 * (90.0 - np.degrees(theta))  # = 180 - 2*theta

    arc_r = r * 0.18
    half  = narrow_deg / 2.0

    angle1_deg = 180.0 - half  # upper-left arm direction
    angle2_deg = 180.0 + half  # lower-left arm direction

    arc_angles = np.linspace(np.radians(angle1_deg), np.radians(angle2_deg), 100)
    arc_x = arc_r * np.cos(arc_angles)
    arc_z = arc_r * np.sin(arc_angles)
    ax.plot(arc_x, arc_z, color=color, lw=lw * 0.8, alpha=0.9)

    # Label at arc midpoint (straight left, 180°)
    label_r = arc_r * 1.4
    label_x = -label_r
    label_z = 0.0

    label_str = rf"${narrow_deg:.1f}^\circ$"

    txt = ax.text(
        label_x, label_z, label_str,
        ha="right", va="center",
        fontsize=FONT_SIZE / 2,
        color=color,
    )
    txt.set_path_effects([
        mpl_pe.withStroke(linewidth=2, foreground="white"),
    ])

    return theta, ax

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
    contours=False,
    radii_save_path=None,
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
        normalize=True,
    )

    if xlim is not None:
        render_kwargs["xlim"] = (-xlim, xlim)
        render_kwargs["ylim"] = (-xlim, xlim)

    if xsec is not None:
        render_kwargs["xsec"] = xsec

    sdf.render(quantity, **render_kwargs)

    if interpolate and fraction < 1.0:
        interpolate_rendered_image(ax)

    if contours and ax.images:
        img = ax.images[0]
        data = np.asarray(img.get_array())

        _, vmax_img = img.get_clim()
        contour_values = [0.01 * np.nanmax(data), 0.5 * np.nanmax(data)]

        valid = [
            cv for cv in contour_values
            if np.nanmin(data) < cv < np.nanmax(data)
        ]
        print("array min/max:", np.nanmin(data), np.nanmax(data))
        print("clim:", img.get_clim())

        for cv in contour_values:
            print("contour:", cv)

        if valid:
            cs = ax.contour(
                data,
                levels=valid,
                colors="white",
                linestyles=["--", "-"],
                linewidths=1.0,
                origin="image",
                extent=img.get_extent(),
            )
            radii = extract_contour_radii(cs)
            for level, r in radii.items():
                print(f"Level {level:.3e}  ->  R (median per path) = {r}")

            draw_radii_arrows(ax, radii, plane=plane)

            if radii_save_path is not None:
                save_contour_radii(
                    radii_save_path, quantity, plane,
                    radii, contour_values,
                )
        else:
            print(f"Skipping contours for {quantity} (fraction={fraction:.2f}): "
                  f"contour levels {contour_values} outside data range "
                  f"[{np.nanmin(data):.2e}, {np.nanmax(data):.2e}]")
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
    contours,
    save_path,
    show,
    col_dens=False,
    radii_save_path=None,
    time_yr=None,
    wind_angle=False,
    v_wind_kms=10.0,
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
        contours=contours,
        radii_save_path=radii_save_path,
    )

    format_render_axes(ax, quantity, plane, meta, xlim=xlim, xsec=xsec, plot_single=True)

    mappable = ax.images[0]
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(quantity_label(quantity, dens_weight, col_dens=col_dens), fontsize=FONT_SIZE / 2)
    cbar.ax.tick_params(labelsize=FONT_SIZE / 2)

    plt.tight_layout()

    if time_yr is not None:
        _add_time_label(fig, time_yr)

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
        frac_label = rf"\textbf{{{frac_pct}\%}}" if frac_pct >= 1 else rf"\textbf{{{frac * 100:.1f}\%}}"
        text = ax.text(
            0.97, 0.97,
            frac_label,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=FONT_SIZE+6,
            color="white",
            fontweight="bold",
        )
        text.set_path_effects([pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()])

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
# Physical parameters overview  (2 rows x 3 cols)
# ===========================================================================
def plot_phys_overview(
    phantom_path: Path,
    dump_path: Path,
    xlim: float | None,
    save_path: Path | None,
    show: bool,
    log_scale: bool = True,
    time_yr = None,
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
    # Layout
    # ------------------------------------------------------------------
    fig.subplots_adjust(
        left=0.07, right=0.99,
        bottom=0.06, top=0.78,
        wspace=-0.3, hspace=0.08,
    )

    fig.canvas.draw()

    # ------------------------------------------------------------------
    # Colorbars above each xy panel
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
            pad=8,
        )

    # Place time label just above the top row of axes
    top_ax_pos = axes[0, 0].get_position()
    time_label_y = top_ax_pos.y1 + 0.075   # small gap above the top row
    time_yr = _read_time_yr(dump_path)
    if time_yr is not None:
        _add_time_label(fig, time_yr, fontsize=20, x=0.06, y=min(time_label_y, 0.98))

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_velocity_components(
    phantom_path: Path,
    dump_path: Path,
    plane: str = "xz",
    xlim: float | None = None,
    fraction: float = 1.0,
    cmap: str = "bwr",
    vsym: float | None = None,
    wind_angle: bool = False,
    v_wind_kms: float = 10.0,
    save_path: Path | None = None,
    show: bool = False,
    time_yr=None,
):
    """
    Render the two in-plane signed velocity components side by side
    (v_x left, v_z right for plane='xz'; v_x left, v_y right for
    plane='xy'), using a diverging blue(negative)/red(positive)
    colormap centered on zero. Optionally overplots the wind-cone
    half-angle X (see compute_wind_cone_angle / draw_wind_cone_lines).
    """
    print(f"\nLoading {phantom_path.name} (velocity components)")
    sdf, sinks = sarracen.read_phantom(phantom_path)

    unit_dist     = sdf._params["udist"]
    unit_time     = sdf._params["utime"]
    unit_velocity = unit_dist / unit_time
    cms_to_kms    = (1 * u.cm / u.s).to(u.km / u.s).value

    comp1 = "x"
    comp2 = "z" if plane == "xz" else "y"

    sdf["v1_kms"] = sdf[f"v{comp1}"].to_numpy() * unit_velocity * cms_to_kms
    sdf["v2_kms"] = sdf[f"v{comp2}"].to_numpy() * unit_velocity * cms_to_kms

    if fraction < 1.0:
        ids  = sdf["iorig"].to_numpy(dtype=np.int64)
        keep = _particle_keep_mask(ids, fraction)
        sdf  = sdf[keep]
        print(f"Retained {keep.sum():,} particles")

    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3), dpi=300, sharex=True, sharey=True,
    )
    fig.tight_layout(rect=[0,0,0.95,1])
    # plt.tight_layout()

    panel_specs = [("v1_kms", comp1, axes[0]), ("v2_kms", comp2, axes[1])]
    theta = None
    if wind_angle:
        theta = compute_wind_cone_angle(sinks, sdf, v_wind_kms=v_wind_kms)

    for col_name, comp, ax in panel_specs:
        vals = sdf[col_name].to_numpy()
        vals = vals[np.isfinite(vals)]

        render_kwargs = dict(
            x="x", y="y" if plane == "xy" else "z",
            cmap=cmap, cbar=False, log_scale=False,
            dens_weight=True, ax=ax, normalize=True,
        )
        if xlim is not None:
            render_kwargs["xlim"] = (-xlim, xlim)
            render_kwargs["ylim"] = (-xlim, xlim)

        sdf.render(col_name, **render_kwargs)
        if col_name == "v1_kms":
            ax.images[0].set_clim(-6, 6)
        else:
            ax.images[0].set_clim(-3, 3)

        format_render_axes(
            ax, "velocity", plane, {"fraction": fraction},
            xlim=xlim, show_title=False, plot_single=True, 
            show_xlabel=True, show_ylabel=True,
        )

        cbar = fig.colorbar(ax.images[0], ax=ax)
        cbar.set_label(rf"$v_{{{comp}}}$ [km s$^{{-1}}$]", fontsize=FONT_SIZE/2)
        cbar.ax.tick_params(labelsize=FONT_SIZE/2)

        tick_size=4
        ax.tick_params(
        axis="both", which="major",
        direction="in", length=tick_size, width=1,
        colors="k", top=True, right=True,
        labelsize=FONT_SIZE/2,
        )
        ax.tick_params( 
            axis="both", which="minor",
            direction="in", length=tick_size/2, width=1,
            colors="k", top=True, right=True,
        )

        if wind_angle:
            theta, ax = draw_wind_cone_lines(ax, theta, xlim=xlim, color="k")

        # if col_name == "v2_kms":
        #     ax.legend()

    if theta is not None:
        print(f"Wind-cone angle theta = {np.degrees(theta):.2f} deg "
              f"(v_wind={v_wind_kms} km/s)")

    if time_yr is not None:
        _add_time_label(fig, time_yr)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_parent_daughter_pair(
    phantom_path,
    dump_path,
    parent,
    daughter,
    plane,
    xlim,
    xsec,
    dens_weight,
    interpolate,
    cmap,
    save_path,
    show,
    col_dens=False,
    time_yr=None,
):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, 4),
        dpi=300,
        sharex=True,
        sharey=True,
    )

    # Parent
    parent_sdf, meta = prepare_render_dataframe(
        phantom_path,
        dump_path,
        parent,
        fraction=1.0,
        col_dens=col_dens,
    )

    render_quantity(
        sdf=parent_sdf,
        quantity=parent,
        plane=plane,
        ax=axes[0],
        xlim=xlim,
        xsec=xsec,
        dens_weight=dens_weight,
        cmap=cmap,
        log_scale=True,
        interpolate=interpolate,
        fraction=1.0,
    )

    # Daughter
    daughter_sdf, _ = prepare_render_dataframe(
        phantom_path,
        dump_path,
        daughter,
        fraction=1.0,
        col_dens=col_dens,
    )

    render_quantity(
        sdf=daughter_sdf,
        quantity=daughter,
        plane=plane,
        ax=axes[1],
        xlim=xlim,
        xsec=xsec,
        dens_weight=dens_weight,
        cmap=cmap,
        log_scale=True,
        interpolate=interpolate,
        fraction=1.0,
    )

    # Match daughter's colour scale to parent
    parent_img   = axes[0].images[0]
    daughter_img = axes[1].images[0]

    parent_vmin, parent_vmax = parent_img.get_clim()
    daughter_img.set_clim(parent_vmin, parent_vmax)

    format_render_axes(
        axes[0], parent, plane, meta,
        xlim=xlim, xsec=xsec,
        q_in_title=False, show_title=False, plot_single=True,
    )

    format_render_axes(
        axes[1], daughter, plane, meta,
        xlim=xlim, xsec=xsec,
        q_in_title=False, show_title=False, plot_single=True,
    )

    axes[0].set_title(format_species_label(parent),   fontsize=FONT_SIZE / 2)
    axes[1].set_title(format_species_label(daughter), fontsize=FONT_SIZE / 2)
    axes[1].set_ylabel('')

    fig.subplots_adjust(
        left=0.08, right=0.88,
        bottom=0.10, top=0.92,
        wspace=0.1,
    )

    fig.canvas.draw()

    left_pos  = axes[0].get_position()
    right_pos = axes[1].get_position()

    cbar_ax = fig.add_axes([
        right_pos.x1 + 0.01,
        left_pos.y0,
        0.02,
        left_pos.y1 - left_pos.y0,
    ])

    cbar = fig.colorbar(parent_img, cax=cbar_ax)
    cbar.set_label(
        quantity_label(parent, dens_weight, col_dens=col_dens),
        fontsize=FONT_SIZE / 2,
    )
    cbar.ax.tick_params(labelsize=FONT_SIZE / 2)

    if time_yr is not None:
        _add_time_label(fig, time_yr)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def _maybe_use_offset_cbar_ticks(cbar, log_scale, font_size):
    """
    Fix colorbar tick labels when the mapped data range spans < 1 order of
    magnitude.  Two distinct failure modes are handled:
    """
    vmin, vmax = cbar.norm.vmin, cbar.norm.vmax
    if vmin is None or vmax is None or vmin <= 0 or vmax <= 0:
        return

    log_span = math.log10(vmax) - math.log10(vmin)
    if log_span >= 1.0:
        return  # ≥ 1 decade: default formatting is fine for both norm types

    # ── shared power-of-ten exponent from the larger bound ─────────────────
    exp    = math.floor(math.log10(vmax))
    factor = 10.0 ** exp

    # Candidate significands; pick those whose product with factor falls in range
    subs  = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ticks = [s * factor for s in subs
             if vmin * 0.999 <= s * factor <= vmax * 1.001]

    if len(ticks) < 2:
        # Extremely tight range: fall back to 3 linearly spaced interior points
        ticks = list(np.linspace(vmin, vmax, 5)[1:-1])

    # set_ticks accepts actual data-space values for both LogNorm and Normalize
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t / factor:.2g}" for t in ticks])

    # Shared multiplier annotation just above the bar
    cbar.ax.text(
        0.5, 1.02,
        rf"$\times10^{{{exp}}}$",
        transform=cbar.ax.transAxes,
        ha="center", va="bottom",
        fontsize=font_size * 0.70,
        clip_on=False,
    )

# ===========================================================================
# Molecule grid  (4-column overview with per-panel colorbars)
# ===========================================================================
def plot_molecule_grid(
    phantom_path,
    dump_path,
    quantities,
    plane,
    xlim,
    xsec,
    dens_weight,
    interpolate,
    log_scale,
    cmap,
    save_path,
    show,
    col_dens=False,
    contours=False,
    radii_save_path=None,
    time_yr=None,
):
    n = len(quantities)
    if n == 0:
        print("plot_molecule_grid: no quantities supplied, skipping.")
        return
    FONT_SIZE = 12

    ncols     = 4
    nrows     = math.ceil(n / ncols)
    panel_w   = 1.9
    cbar_w_in = 0.18
    gap_w_in  = 0.45
    fig_w     = ncols * (panel_w + cbar_w_in + gap_w_in)
    fig_h     = nrows * panel_w

    col_ratios = [panel_w, cbar_w_in, gap_w_in] * ncols

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs  = fig.add_gridspec(
        nrows, ncols * 3,
        width_ratios=col_ratios,
        wspace=0.0,
        hspace=0.25,   # ← was 0.15; extra room for the ×10^N annotation
        left=0.09, right=0.98,
        bottom=0.08, top=0.97,
    )

    shared_ax = None

    for i, quantity in enumerate(quantities):
        print(f"Rendering {quantity} ({i+1}/{n})")
        row    = i // ncols
        col    = i % ncols
        gs_col  = col * 3
        gs_cbar = gs_col + 1

        kwargs = dict(sharex=shared_ax, sharey=shared_ax) if shared_ax else {}
        ax  = fig.add_subplot(gs[row, gs_col], **kwargs)
        cax = fig.add_subplot(gs[row, gs_cbar])

        if shared_ax is None:
            shared_ax = ax

        is_bottom_in_col = (i + ncols) >= n
        is_leftmost      = (col == 0)

        sdf, meta = prepare_render_dataframe(
            phantom_path, dump_path, quantity,
            fraction=1.0, col_dens=col_dens,
        )

        render_quantity(
            sdf=sdf, quantity=quantity, plane=plane, ax=ax,
            xlim=xlim, xsec=xsec, dens_weight=dens_weight,
            cmap=cmap, log_scale=log_scale, interpolate=interpolate,
            fraction=1.0, contours=contours, radii_save_path=radii_save_path,
        )

        format_render_axes(
            ax, quantity, plane, meta,
            xlim=xlim, xsec=xsec,
            show_xlabel=is_bottom_in_col,
            show_ylabel=is_leftmost,
            small_ticks=True,
            plot_single=True,
        )

        if not is_bottom_in_col:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
        if not is_leftmost:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")

        mappable = ax.images[0]
        cbar = fig.colorbar(mappable, cax=cax)

        cbar.ax.tick_params(
            labelsize=FONT_SIZE * 0.7,
            direction="out",
            length=2,
            width=0.6,
        )
        if col == ncols - 1 or i == n - 1:
            cbar.set_label(
                quantity_label(quantity, dens_weight, col_dens=col_dens),
                fontsize=FONT_SIZE * 0.75,
            )
        cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

        # ── compact tick labels for narrow-range colorbars ─────────────────
        # Must come after set_label so the helper can read any existing ylabel.
        _maybe_use_offset_cbar_ticks(cbar, log_scale, FONT_SIZE)
        # ───────────────────────────────────────────────────────────────────

        ax.set_title(ax.get_title(), y=0.98)
    
    if time_yr is not None:
        _add_time_label(fig, time_yr, x=0.03, y=0.995)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ===========================================================================
# Render config reader
# ===========================================================================
import csv


def _parse_bool(val: str) -> bool:
    """Return True for 'TRUE'/'True'/'1'/'yes'; False otherwise."""
    return str(val).strip().lower() in ("true", "1", "yes")


def _parse_float_or_none(val: str):
    """Return float(val) or None for empty / 'None' strings."""
    v = val.strip()
    return None if v in ("", "None") else float(v)


def load_render_config_row(path: str, row_index: int) -> dict:
    """
    Read one row (0-based) from *configs_render.csv* and return a dict
    with the same keys used by the argparse namespace in main().

    CSV columns
    -----------
    quantity        : molecule name, 'parents', 'daughters', 'molecules', or
                      empty (phys-overview / parent-daughter infer their own)
    plane           : 'xy', 'xz', or 'xy,xz'  (comma-separated)
    xlim            : one or more AU values, comma-separated (or empty)
    dens_weight     : TRUE / FALSE
    col-dens        : TRUE / FALSE
    mol-grid        : TRUE / FALSE
    compare_fractions : TRUE / FALSE
    nine-fractions  : TRUE / FALSE
    xsec            : float or empty
    contours        : TRUE / FALSE
    parent-daughter : TRUE / FALSE
    phys-overview   : TRUE / FALSE
    """
    molecules = ['CO', 'CH2', 'CH3', 'CH4', 'HCl', 'CH3CN', 'SiO', 'HCN',
                'CN', 'HC3N', 'HC5N', 'HC7N', 'C2H', 'C4H', 'C6H', 'SiC',
                'SiN', 'H2CS', 'H2CO', 'N2', 'NH3', 'H2S', 'HCP', 'H2O',
                'C2H2', 'CS', 'SiC2', 'HF', 'C2H4', 'SiS']
    parents_list = ["CO", "N2", "CH4", "H2O", "SiC2", "CS", "C2H2",
                    "HCN", "SiS", "SiO", "HCl", "C2H4", "NH3", "HCP",
                    "HF", "H2S"]
    daughters_list = ['CH2', 'CH3', 'CH3CN', 'CN', 'HC3N', 'HC5N', 'HC7N',
                      'C2H', 'C4H', 'C6H', 'SiC', 'SiN', 'H2CS', 'H2CO',
                      'SO', 'SO2']

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(
            f"--config-row {row_index} is out of range; "
            f"{path} has {len(rows)} data rows (0-{len(rows)-1})."
        )

    row = rows[row_index]

    # ---- quantity -----------------------------------------------------------
    q_raw = row.get("quantity", "").strip()
    if q_raw == "parents":
        quantity = parents_list
    elif q_raw == "daughters":
        quantity = daughters_list
    elif q_raw == "molecules":
        quantity = molecules
    elif q_raw == "":
        quantity = []          # phys-overview / parent-daughter handle their own
    else:
        quantity = [q_raw]     # single named species, e.g. 'CO'

    # ---- plane --------------------------------------------------------------
    plane = [p.strip() for p in row.get("plane", "xy").split(",") if p.strip()]

    # ---- xlim  (one or more values) -----------------------------------------
    xlim_raw = row.get("xlim", "").strip()
    if xlim_raw in ("", "None"):
        xlim = [None]
    else:
        xlim = [_parse_float_or_none(v) for v in xlim_raw.split(",")]

    # ---- xsec ---------------------------------------------------------------
    xsec_raw = row.get("xsec", "").strip()
    if xsec_raw in ("", "None"):
        xsec = [None]
    else:
        xsec = [_parse_float_or_none(v) for v in xsec_raw.split(",")]

    # If planes and xlims differ in length, zip will stop at the shorter one;
    # that mirrors how the normal CLI --plane / --xlim / --xsec work.

    return dict(
        quantity          = quantity,
        plane             = plane,
        xlim              = xlim,
        xsec              = xsec,
        dens_weight       = _parse_bool(row.get("dens_weight", "FALSE")),
        col_dens          = _parse_bool(row.get("col-dens", "FALSE")),
        mol_grid          = _parse_bool(row.get("mol-grid", "FALSE")),
        compare_fractions = _parse_bool(row.get("compare_fractions", "FALSE")),
        nine_fractions    = _parse_bool(row.get("nine-fractions", "FALSE")),
        contours          = _parse_bool(row.get("contours", "FALSE")),
        parent_daughter   = _parse_bool(row.get("parent-daughter", "FALSE")),
        phys_overview     = _parse_bool(row.get("phys-overview", "FALSE")),
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
    parser.add_argument(
        "--contours",
        action="store_true",
        help="Draw white dashed contours at 0.5x and 0.01x of the colourbar maximum.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dark-mode", action="store_true")
    parser.add_argument("--chemistry", choices=["Crich", "Orich"], default="Crich")
    parser.add_argument("--col-dens", action="store_true", default=False)
    parser.add_argument(
        "--phys-overview",
        action="store_true",
        help=(
            "Render a 2x3 overview panel of all three physical parameters "
            "(temperature, density, Av) in xy and xz projections.  "
            "Only --xlim is needed alongside --dump / --dump-index."
        ),
    )
    parser.add_argument(
        "--velocity-components",
        action="store_true",
        help=(
            "Render v_x (left) and v_z (or v_y for plane=xy) (right) "
            "side by side with a diverging (signed) colormap. "
            "Combine with --wind-angle to overplot the cone X."
        ),
    )
    parser.add_argument(
        "--vsym", type=float, default=None,
        help="Symmetric color limit [+/-vsym km/s] for --velocity-components "
             "(default: 99th percentile of |v|).",
    )
    parser.add_argument("--wind-angle", action="store_true",
        help="Overplot a line at theta = arctan(v_wind/|velAGB|) from the AGB's direction of motion.")
    parser.add_argument("--v-wind", type=float, default=10.0,
        help="Wind speed in km/s used for --wind-angle (default: 10).")
    parser.add_argument(
        "--radii-save-path",
        type=str,
        default=None,
        help="Path to a .txt file where contour radii are saved/appended.",
    )
    parser.add_argument(
        "--parent-daughter",
        action="store_true",
        help="Plot predefined parent/daughter chemistry pairs."
    )
    parser.add_argument(
        "--mol-grid",
        action="store_true",
        help=(
            "Render all --quantity values together as a 4-column grid, each "
            "panel with its own colorbar and shared spatial axes.  Saved "
            "directly to <base>/render/ as a final figure.  Incompatible "
            "with --compare-fractions and --parent-daughter."
        ),
    )
    # --- SLURM / CSV-batch arguments -----------------------------------------
    parser.add_argument(
        "--config-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to configs_render.csv.  When supplied together with "
            "--config-row the script reads one row and uses it to override "
            "all render-mode flags (quantity, plane, xlim, xsec, "
            "dens-weight, col-dens, mol-grid, compare-fractions, "
            "nine-fractions, contours, parent-daughter, phys-overview)."
        ),
    )
    parser.add_argument(
        "--config-row",
        type=int,
        default=None,
        metavar="N",
        help="0-based row index into --config-csv (set to SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--config-plane",
        type=str,
        default=None,
        metavar="PLANE",
        help="Override: restrict this task to a single plane (xy or xz).",
    )
    parser.add_argument(
        "--config-xlim",
        type=str,
        default=None,
        metavar="XLIM",
        help="Override: restrict this task to a single xlim value (float or 'None').",
    )
    parser.add_argument(
        "--config-xsec",
        type=str,
        default=None,
        metavar="XSEC",
        help="Override: restrict this task to a single xsec value (float or 'None').",
    )

    molecules = ['CO', 'CH2', 'CH3', 'CH4', 'HCl', 'CH3CN', 'SiO', 'HCN',
                'CN', 'HC3N', 'HC5N', 'HC7N', 'C2H', 'C4H', 'C6H', 'SiC',
                'SiN', 'H2CS', 'H2CO', 'N2', 'NH3', 'H2S', 'HCP', 'H2O', 'C2H2',
                'CS', 'SiC2', 'HF', 'C2H4', 'SiS']
    parents = ["CO", "N2", "CH4", "H2O", "SiC2", "CS", "C2H2",
               "HCN", "SiS", "SiO", "HCl", "C2H4", "NH3", "HCP", "HF", "H2S"]
    daughters = ['CH2', 'CH3', 'CH3CN', 'CN', 'HC3N', 'HC5N', 'HC7N',
                'C2H', 'C4H', 'C6H', 'SiC', 'SiN', 'H2CS', 'H2CO', 'SO', 'SO2']

    PARENT_DAUGHTER_PAIRS = [
        ("C2H2", "C2H"),
        ("HCN",  "CN"),
    ]

    if is_interactive():
        args = parser.parse_args([
            # "--dump-index", "1190",
            "--dump-index", "1581",
            "--plane", "xz",
            # "--quantity", "velocity",
            "--xlim", "750",
            # "--phys-overview",
            "--velocity-components",
            "--wind-angle",
            "--v-wind", "10.0",
            # "--mol-grid",
            # "--parent-daughter",
            # "--dens-weight", "True",
            # "--vmin", "1e-8",
            # "--vmax", "1e-4",
            # "--contours",
            # "--xsec", "0",
            # "--log", "False",
            # "--compare-fractions",
            # "--nine-fractions",
            # "--interpolate",
            # "--col-dens",
            # "--radii-save-path", "/fred/oz304/beckers/v10a09_out/output/radii_contours.txt",
            "--show",
            # "--dark-mode",
        ])
    else:
        args = parser.parse_args()

    # ---- CSV row override (used by SLURM array jobs) ------------------------
    if args.config_csv is not None and args.config_row is not None:
        cfg = load_render_config_row(args.config_csv, args.config_row)
        print(f"[config-row {args.config_row}] loaded from {args.config_csv}:")
        for k, v in cfg.items():
            print(f"  {k:22s} = {v}")
        # Override args with values from the CSV row
        args.quantity          = cfg["quantity"]
        args.plane             = cfg["plane"]
        args.xlim              = cfg["xlim"]
        args.xsec              = cfg["xsec"]
        args.dens_weight       = cfg["dens_weight"]
        args.col_dens          = cfg["col_dens"]
        args.mol_grid          = cfg["mol_grid"]
        args.compare_fractions = cfg["compare_fractions"]
        args.nine_fractions    = cfg["nine_fractions"]
        args.contours          = cfg["contours"]
        args.parent_daughter   = cfg["parent_daughter"]
        args.phys_overview     = cfg["phys_overview"]

    # ---- Per-task plane/xlim/xsec overrides (set by SLURM from task_list) ---
    # These narrow a multi-value CSV row to exactly one combination so that
    # each array task renders only its assigned (plane, xlim, xsec).
    if args.config_plane is not None:
        args.plane = [args.config_plane]
    if args.config_xlim is not None:
        args.xlim = [None if args.config_xlim.strip() == "None"
                     else float(args.config_xlim)]
    if args.config_xsec is not None:
        args.xsec = [None if args.config_xsec.strip() == "None"
                     else float(args.config_xsec)]
    # -------------------------------------------------------------------------

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

    def _save_dir(quantity: str, xlim: float | None, compare: bool, 
                  overview: bool = False, parent_daughter: bool = False):
        mol_type = "parent" if quantity in parents else "daughter" if quantity in daughters else "other"
        spatial_tag = "zoom" if (xlim is not None and xlim <= 100) else "full"
        base = SAVE_DIR_BASE / args.chemistry / "render"
        if overview:
            d = base / "phys_overview"
        elif parent_daughter:
            d = base / "parent_daughter" / spatial_tag
        elif compare:
            d = base / "compare" / mol_type / spatial_tag
        else:
            d = base / "single" / mol_type / spatial_tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Physical overview mode — ignores --quantity / --plane / --compare
    # ------------------------------------------------------------------
    if args.phys_overview:
        xlim = args.xlim[0]
        save_dir  = _save_dir(None, xlim, compare=False, overview=True)
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
    
    if args.velocity_components:
        for plane, xlim, xsec in product(args.plane, args.xlim, args.xsec):
            xlim_tag = f"_{xlim:.0f}AU" if xlim is not None else ""
            save_dir = SAVE_DIR_BASE / args.chemistry / "render" / "velocity_components"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"velcomp_{plane}_{dump_stem}{xlim_tag}.{save_ext}"

            plot_velocity_components(
                phantom_path=phantom_path,
                dump_path=dump_path,
                plane=plane,
                xlim=xlim,
                fraction=args.fraction,
                vsym=args.vsym,
                wind_angle=args.wind_angle,
                v_wind_kms=args.v_wind,
                save_path=save_path,
                show=args.show,
            )
        return
    # ------------------------------------------------------------------
    # Parent / daughter comparison mode
    # ------------------------------------------------------------------
    if args.parent_daughter:

        for plane, xlim, xsec in product(args.plane, args.xlim, args.xsec):

            for parent, daughter in PARENT_DAUGHTER_PAIRS:

                xsec_tag = f"_xsec{xsec:.0f}" if xsec is not None else ""
                xlim_tag = f"_{xlim:.0f}AU" if xlim is not None else ""

                save_dir = _save_dir(None, xlim, compare=False, overview=False, parent_daughter=True)

                save_path = save_dir / (
                    f"render_{plane}_{parent}_{daughter}_"
                    f"{dump_stem}"
                    f"{xsec_tag}"
                    f"{xlim_tag}.{save_ext}"
                )

                plot_parent_daughter_pair(
                    phantom_path=phantom_path,
                    dump_path=dump_path,
                    parent=parent,
                    daughter=daughter,
                    plane=plane,
                    xlim=xlim,
                    xsec=xsec,
                    dens_weight=args.dens_weight,
                    interpolate=args.interpolate,
                    cmap=args.cmap,
                    save_path=save_path,
                    show=args.show,
                    col_dens=args.col_dens,
                )

        return

    # ------------------------------------------------------------------
    # Molecule grid mode — all quantities in one 4-column figure
    # ------------------------------------------------------------------
    if args.mol_grid:
        for plane, xlim, xsec in product(args.plane, args.xlim, args.xsec):
                    xsec_tag = f"_xsec{xsec:.0f}" if xsec is not None else ""
                    xlim_tag = f"_{xlim:.0f}AU"   if xlim is not None else ""
                    dw_tag   = "_dw"              if args.dens_weight  else ""
                    cd_tag   = "n"                if args.col_dens     else ""

                    # Label the file by which canonical list was passed, or fall back
                    # to a generic tag
                    if all(q in parents for q in args.quantity):
                        mol_tag = "parents"
                    elif all(q in daughters for q in args.quantity):
                        mol_tag = "daughters"
                    else:
                        mol_tag = "molecules"

                    save_dir = SAVE_DIR_BASE / args.chemistry / "render" / 'grid' / mol_tag
                    save_dir.mkdir(parents=True, exist_ok=True)

                    save_path = save_dir / (
                        f"grid_{plane}_{cd_tag}{mol_tag}_{dump_stem}"
                        f"{xsec_tag}{dw_tag}{xlim_tag}.{save_ext}"
                    )

                    plot_molecule_grid(
                        phantom_path=phantom_path,
                        dump_path=dump_path,
                        quantities=args.quantity,
                        plane=plane,
                        xlim=xlim,
                        xsec=xsec,
                        dens_weight=args.dens_weight,
                        interpolate=args.interpolate,
                        log_scale=args.log_scale,
                        cmap=args.cmap,
                        save_path=save_path,
                        show=args.show,
                        col_dens=args.col_dens,
                        contours=args.contours,
                        radii_save_path=Path(args.radii_save_path) if args.radii_save_path else None,
                    )

        return

    # ------------------------------------------------------------------
    # Normal (single / compare-fractions) mode
    # ------------------------------------------------------------------
    fractions = FRACTIONS_9 if args.nine_fractions else FRACTIONS_3
    cd_tag    = "n" if args.col_dens else ""

    for quantity in args.quantity:
        for plane, xlim, xsec in product(args.plane, args.xlim, args.xsec):
            xsec_tag  = f"_xsec{xsec:.0f}"  if xsec is not None else ""
            dw_tag    = "_dw"                if args.dens_weight      else ""
            xlim_tag  = f"_{xlim:.0f}AU"    if xlim is not None else ""

            if args.compare_fractions:
                frac_set_tag = "_frac9" if args.nine_fractions else "_frac3"
                save_dir  = _save_dir(quantity, xlim, compare=True)
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
                save_dir  = _save_dir(quantity, xlim, compare=False)
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
                    contours=args.contours,
                    save_path=save_path,
                    show=args.show,
                    col_dens=args.col_dens,
                    radii_save_path=Path(args.radii_save_path) if args.radii_save_path else None,
                    wind_angle=args.wind_angle, v_wind_kms=args.v_wind,
                )


if __name__ == "__main__":
    main()