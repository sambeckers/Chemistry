#!/usr/bin/env python3
"""
plot_stats.py

Computes mean molecular abundances vs radius by accumulating ALL HDF5 dump
files from aphid, then plots those 3-D averages against the 1-D chemistry model.
Also averages and plots physical parameters (temperature, AV, density).

NEW: Particle subsampling mode (--mode plot-fraction-compare) allows testing
the effect of using only 10% or 50% of particles (random, deterministic) on
the derived abundance profiles.
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
from code_chem.plotting.plot_utils import apply_abundance_axis_limits, add_log_ticks, _collect_xy


# ---------------------------------------------------------------------------
# Configuration — edit to match your setup
# ---------------------------------------------------------------------------

DUMP_DIR    = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
SCRATCH_DIR = Path("/aphid/scratch-3month/sbeckers/v10a09_out/accum_scratch")
SAVE_DIR_BASE = BASE_PATH / "figures/v10a09_out"

# Radius domain (cm)
R_MIN = 1e13
R_MAX = 1e17
N_BINS = 300                  # log-spaced radial bins

# Abundance histogram for percentile estimation
# 150 log-spaced bins covering the physically meaningful range.
# Extend AB_LOG_MIN further negative if your species reach lower abundances.
N_AB_BINS   = 150
AB_LOG_MIN  = -20.0           # log10 of the lowest abundance edge
AB_LOG_MAX  =  0.0            # log10 of the highest abundance edge (= 1)
AB_EDGES    = np.logspace(AB_LOG_MIN, AB_LOG_MAX, N_AB_BINS + 1)
AB_CENTRES  = np.sqrt(AB_EDGES[:-1] * AB_EDGES[1:])   # geometric centres

# ---------------------------------------------------------------------------
# Physical parameter configuration
# ---------------------------------------------------------------------------
# Each entry defines the HDF5 dataset name, histogram bin edges (log-spaced),
# axis label, and display colour for the single-panel plot.
# Adjust log_min/log_max if your simulations cover a different dynamic range.

PHYS_PARAMS: dict[str, dict] = {
    # Observed range from a data subset is given first; edges are then
    # extended by ~1–2 dex on each side so histogram tails are captured
    # even if the full simulation is more extreme than the subset.
    "temperature": {
        "hdf5_key":  "temp",           # particles/<hdf5_key>
        # observed: 10^1 – 10^5 K  →  extend to 10^0 – 10^6 K
        "log_min":   0.0,                     # 1 K
        "log_max":   6.0,                     # 1 000 000 K
        "n_bins":    150,
        "label":     "Temperature [K]",
        "color":     "#d6604d",
    },
    "av": {
        "hdf5_key":  "av",
        # observed: 10^-4 – 10^-1 mag  →  extend to 10^-6 – 10^1 mag
        "log_min":   -6.0,                    # 10^-6 mag
        "log_max":    1.0,                    # 10 mag
        "n_bins":    150,
        "label":     r"$A_V$ [mag]",
        "color":     "#4dac26",
    },
    "density": {
        "hdf5_key":  "density",
        # observed: 10^0 – 10^8 cm^-3  →  extend to 10^-1 – 10^10 cm^-3
        "log_min":   -1.0,                    # 0.1 cm⁻³
        "log_max":   10.0,                    # 10^10 cm⁻³
        "n_bins":    150,
        "label":     r"Density [$n$, cm$^{-3}$]",
        "color":     "#762a83",
    },
}

# Pre-compute edges and geometric centres for every physical parameter.
for _cfg in PHYS_PARAMS.values():
    _cfg["edges"]   = np.logspace(_cfg["log_min"], _cfg["log_max"],
                                  _cfg["n_bins"] + 1)
    _cfg["centres"] = np.sqrt(_cfg["edges"][:-1] * _cfg["edges"][1:])

# Gaussian smooth width in bins (log-spaced); set 0 to disable
SMOOTH_SIGMA = 3

# Bins with fewer particles than this are masked as NaN
MIN_COUNT = 30

# Threads per SLURM task
MAX_WORKERS = 32

# Percentile pair used for the shaded uncertainty band
PERCENTILE_LO = 16
PERCENTILE_HI = 84


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

SPECIES_HDF5_KEY: dict[str, str] = {sp: normalise_name(sp) for sp in get_all_species()}

def get_species_and_colors(chemistry: str, present: set[str]):
    """Return (parent_list, daughter_list, color_dict) for species in *present*."""
    daughters = daughters_Orich if chemistry == "Orich" else daughters_Crich
    parent_species   = [s for s in parents   if s in present]
    daughter_species = [s for s in daughters if s in present]
    all_species = parent_species + [s for s in daughter_species if s not in parent_species]
    cmap = generate_colormap(len(all_species)).colors
    colors = {sp: cmap[i] for i, sp in enumerate(all_species)}
    return parent_species, daughter_species, colors


# ---------------------------------------------------------------------------
# Deterministic particle filter  (vectorised)
# ---------------------------------------------------------------------------

def _particle_keep_mask(
    ids: np.ndarray,
    fraction: float,
    seed: int = 123456789,
) -> np.ndarray:
    """
    Return a boolean mask selecting a deterministic ~*fraction* of particles.

    The same particle ID always produces the same decision for a given
    (fraction, seed) pair, so a particle present in multiple dumps is
    consistently included or excluded across all of them.

    Parameters
    ----------
    ids : np.ndarray of integer dtype
        Unique particle IDs for one dump.
    fraction : float
        Target inclusion fraction in [0, 1].
    seed : int
        Change to draw a different reproducible sample from the same population.

    Returns
    -------
    np.ndarray of bool, same length as *ids*.
    """
    if fraction >= 1.0 - 1e-9:
        return np.ones(len(ids), dtype=bool)
    if fraction <= 1e-9:
        return np.zeros(len(ids), dtype=bool)

    # XOR the seed in so different seeds give different samples, then apply a
    # Knuth multiplicative hash to spread sequential IDs across [0, 2^64).
    # A particle is kept if its hash value falls in the lowest `fraction` of
    # that range — equivalent to drawing a uniform float in [0, 1) and
    # checking < fraction, but without any floating-point arithmetic.
    h = (ids.astype(np.uint64) ^ np.uint64(seed)) * np.uint64(0x9E3779B97F4A7C15)
    return h < np.uint64(int(fraction * 0xFFFF_FFFF_FFFF_FFFF))


# ---------------------------------------------------------------------------
# Per-dump worker
# ---------------------------------------------------------------------------

def _accumulate_dump(path: Path, species_list: list[str], fraction: float = 1.0):
    """
    Read r, particle IDs, every requested species, and physical parameters from
    one HDF5 dump. Particles are filtered according to `fraction` using their
    unique ID (deterministic). For fraction=1.0, all particles are kept.
    """
    import h5py

    species_result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    phys_result:    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    try:
        with h5py.File(path, "r") as f:
            r = f["particles/r"][:]

            # Only read IDs and compute a keep-mask when actually subsampling.
            # For fraction=1.0 every particle is included, so skip both.
            if fraction < 1.0 - 1e-9:
                keep_mask = _particle_keep_mask(f["particles/id"][:], fraction)
            else:
                keep_mask = None

            # --- Chemical species ---
            for species in species_list:
                try:
                    ab = f[f"particles/{SPECIES_HDF5_KEY[species]}"][:]
                except KeyError:
                    continue

                # Mask unphysical values + optional particle filter
                mask = (
                    np.isfinite(r)  &
                    np.isfinite(ab) &
                    (r  > 0)        &
                    (ab > 0)        &
                    (ab <= 1.0)
                )
                if keep_mask is not None:
                    mask &= keep_mask
                n_unphysical = (ab > 1.0).sum()
                if n_unphysical > 0:
                    n_total = np.isfinite(ab).sum()
                    print(f"  {path.name} [{species}]: "
                          f"masked {n_unphysical}/{n_total} unphysical values")

                r_ok, ab_ok = r[mask], ab[mask]
                if r_ok.size == 0:
                    continue

                idx   = np.digitize(r_ok, R_EDGES) - 1
                valid = (idx >= 0) & (idx < N_BINS)
                bin_sum   = np.bincount(idx[valid], weights=ab_ok[valid],
                                        minlength=N_BINS)
                bin_count = np.bincount(idx[valid],
                                        minlength=N_BINS).astype(np.int64)
                hist_2d, _, _ = np.histogram2d(
                    r_ok, ab_ok,
                    bins=[R_EDGES, AB_EDGES],
                )
                species_result[species] = (bin_sum, bin_count,
                                           hist_2d.astype(np.int32))

            # --- Physical parameters ---
            for param, cfg in PHYS_PARAMS.items():
                try:
                    vals = f[f"particles/{cfg['hdf5_key']}"][:]
                except KeyError:
                    continue

                mask = (
                    np.isfinite(r)    &
                    np.isfinite(vals) &
                    (r    > 0)        &
                    (vals > 0)
                )
                if keep_mask is not None:
                    mask &= keep_mask
                r_ok, v_ok = r[mask], vals[mask]
                if r_ok.size == 0:
                    continue

                idx   = np.digitize(r_ok, R_EDGES) - 1
                valid = (idx >= 0) & (idx < N_BINS)
                bin_sum   = np.bincount(idx[valid], weights=v_ok[valid],
                                        minlength=N_BINS)
                bin_count = np.bincount(idx[valid],
                                        minlength=N_BINS).astype(np.int64)
                hist_2d, _, _ = np.histogram2d(
                    r_ok, v_ok,
                    bins=[R_EDGES, cfg["edges"]],
                )
                phys_result[param] = (bin_sum, bin_count,
                                      hist_2d.astype(np.int32))

    except Exception as exc:
        print(f"  Warning: skipping {path.name}: {exc}")

    return species_result, phys_result


# ---------------------------------------------------------------------------
# Chunk accumulation (one SLURM task)
# ---------------------------------------------------------------------------

def accumulate_chunk(
    dump_files: list[Path],
    species_list: list[str],
    max_workers: int = MAX_WORKERS,
    fraction: float = 1.0,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """
    Accumulate bin sums/counts/histograms over *dump_files* using a thread pool,
    filtering particles according to `fraction`.

    Returns
    -------
    species_totals : dict[species -> (sum, count, hist_2d)]
    phys_totals    : dict[param  -> (sum, count, hist_2d)]
    """
    species_totals: dict[str, list] = {
        s: [
            np.zeros(N_BINS),
            np.zeros(N_BINS, dtype=np.int64),
            np.zeros((N_BINS, N_AB_BINS), dtype=np.int64),
        ]
        for s in species_list
    }
    phys_totals: dict[str, list] = {
        p: [
            np.zeros(N_BINS),
            np.zeros(N_BINS, dtype=np.int64),
            np.zeros((N_BINS, cfg["n_bins"]), dtype=np.int64),
        ]
        for p, cfg in PHYS_PARAMS.items()
    }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_accumulate_dump, p, species_list, fraction): p
                   for p in dump_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Dumps"):
            sp_res, phys_res = fut.result()
            for sp, (s, c, h) in sp_res.items():
                species_totals[sp][0] += s
                species_totals[sp][1] += c
                species_totals[sp][2] += h
            for param, (s, c, h) in phys_res.items():
                phys_totals[param][0] += s
                phys_totals[param][1] += c
                phys_totals[param][2] += h

    species_out = {sp: (species_totals[sp][0],
                        species_totals[sp][1],
                        species_totals[sp][2])
                   for sp in species_list}
    phys_out    = {p:  (phys_totals[p][0],
                        phys_totals[p][1],
                        phys_totals[p][2])
                   for p in PHYS_PARAMS}
    return species_out, phys_out


# ---------------------------------------------------------------------------
# SLURM mode: accumulate (now with fraction)
# ---------------------------------------------------------------------------

def run_accumulate(task_id: int, n_tasks: int, chemistry: str, fraction: float = 1.0) -> None:
    """Process this task's slice of dump files, filter particles by fraction,
    and write a partial .npz with fraction encoded in the filename."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    dump_files = sorted(DUMP_DIR.glob("dump_*.h5"))
    if not dump_files:
        raise FileNotFoundError(f"No dump_*.h5 files found in {DUMP_DIR}")

    chunk = dump_files[task_id::n_tasks]
    print(f"Task {task_id + 1}/{n_tasks}: "
          f"processing {len(chunk)}/{len(dump_files)} dumps, "
          f"fraction = {fraction:.1%}")

    species_list = get_all_species()
    species_totals, phys_totals = accumulate_chunk(chunk, species_list, fraction=fraction)

    # Fraction=1.0 uses the original filename format (no fraction suffix)
    # so existing files remain valid without re-running the accumulate step.
    if fraction >= 1.0 - 1e-9:
        out_path = (SCRATCH_DIR /
                    f"partial_{chemistry}_{task_id:04d}_of_{n_tasks:04d}.npz")
    else:
        frac_int = int(round(fraction * 100))
        out_path = (SCRATCH_DIR /
                    f"partial_{chemistry}_f{frac_int:03d}_{task_id:04d}_of_{n_tasks:04d}.npz")
    save_dict: dict[str, np.ndarray] = {}

    # Species data — keyed as  "<species>__sum" / "__count" / "__hist"
    for sp, (s, c, h) in species_totals.items():
        save_dict[f"{sp}__sum"]   = s
        save_dict[f"{sp}__count"] = c
        save_dict[f"{sp}__hist"]  = h

    # Physical parameter data — keyed as  "phys__<param>__sum" etc.
    for param, (s, c, h) in phys_totals.items():
        save_dict[f"phys__{param}__sum"]   = s
        save_dict[f"phys__{param}__count"] = c
        save_dict[f"phys__{param}__hist"]  = h

    np.savez(out_path, **save_dict)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Aggregation: merge all partial .npz files for a given chemistry and fraction
# ---------------------------------------------------------------------------

def aggregate_partials(
    chemistry: str,
    n_tasks: int,
    fraction: float = 1.0,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """
    Aggregate all partial .npz files that match the given chemistry and fraction.

    Returns
    -------
    species_totals : dict[species -> (bin_sum, bin_count, hist_2d)]
        hist_2d shape: (N_BINS, N_AB_BINS)
    phys_totals    : dict[param  -> (bin_sum, bin_count, hist_2d)]
        hist_2d shape: (N_BINS, cfg["n_bins"])
    """
    # Fraction=1.0 matches the original filename format (no fraction tag).
    # Fractional runs use the f<pct> prefix added in run_accumulate.
    if fraction >= 1.0 - 1e-9:
        # Match e.g. partial_Crich_0000_of_0032.npz (4-digit task ID, no f-prefix)
        pattern = f"partial_{chemistry}_[0-9][0-9][0-9][0-9]_of_{n_tasks:04d}.npz"
    else:
        frac_int = int(round(fraction * 100))
        pattern = f"partial_{chemistry}_f{frac_int:03d}_????_of_{n_tasks:04d}.npz"
    files   = sorted(SCRATCH_DIR.glob(pattern))
    if not files:
        if fraction >= 1.0 - 1e-9:
            raise FileNotFoundError(
                f"No partial files matching '{pattern}' in {SCRATCH_DIR}\n"
                f"Run the accumulate step first (fraction=1.0 uses the original "
                f"filename format without a fraction suffix)."
            )
        else:
            raise FileNotFoundError(
                f"No partial files matching '{pattern}' in {SCRATCH_DIR}\n"
                f"Run the accumulate step with --fraction {fraction} first."
            )
    print(f"Aggregating {len(files)} partial files for fraction {fraction:.1%} …")

    species_totals: dict[str, list] = {}
    phys_totals:    dict[str, list] = {}

    for path in files:
        data = np.load(path)
        for key in data.files:
            if key.startswith("phys__"):
                # Physical parameter key: "phys__<param>__<kind>"
                _, param, kind = key.split("__", 2)
                if param not in phys_totals:
                    cfg = PHYS_PARAMS[param]
                    phys_totals[param] = [
                        np.zeros(N_BINS),
                        np.zeros(N_BINS, dtype=np.int64),
                        np.zeros((N_BINS, cfg["n_bins"]), dtype=np.int64),
                    ]
                if kind == "sum":
                    phys_totals[param][0] += data[key]
                elif kind == "count":
                    phys_totals[param][1] += data[key]
                elif kind == "hist":
                    phys_totals[param][2] += data[key]
            else:
                # Species key: "<species>__<kind>"
                sp, kind = key.rsplit("__", 1)
                if sp not in species_totals:
                    species_totals[sp] = [
                        np.zeros(N_BINS),
                        np.zeros(N_BINS, dtype=np.int64),
                        np.zeros((N_BINS, N_AB_BINS), dtype=np.int64),
                    ]
                if kind == "sum":
                    species_totals[sp][0] += data[key]
                elif kind == "count":
                    species_totals[sp][1] += data[key]
                elif kind == "hist":
                    species_totals[sp][2] += data[key]

    sp_out   = {sp: (v[0], v[1], v[2]) for sp, v in species_totals.items()}
    phys_out = {p:  (v[0], v[1], v[2]) for p,  v in phys_totals.items()}
    return sp_out, phys_out


# ---------------------------------------------------------------------------
# Smoothing (unchanged)
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
    result = np.where(result > 1.0, np.nan, result)
    return result


def smooth_mean_phys(
    bin_sum: np.ndarray,
    bin_count: np.ndarray,
    min_count: int = MIN_COUNT,
    sigma: float = SMOOTH_SIGMA,
) -> np.ndarray:
    """Same log-space Gaussian smoothing as smooth_mean, but without the
    abundance <= 1 upper-bound clipping that is only meaningful for
    fractional abundances."""
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
    return 10.0 ** log_smoothed


def compute_percentile_bands(
    hist_2d: np.ndarray,
    bin_centres: np.ndarray,
    percentiles: list[int] | None = None,
    min_count: int = MIN_COUNT,
) -> dict[int, np.ndarray]:
    """Derive per-radial-bin percentiles from an aggregated 2-D histogram."""
    if percentiles is None:
        percentiles = [PERCENTILE_LO, 50, PERCENTILE_HI]

    n_val_bins  = hist_2d.shape[1]
    row_counts  = hist_2d.sum(axis=1)
    cdf         = np.cumsum(hist_2d.astype(np.float64), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(row_counts[:, None] >= min_count,
                        cdf / row_counts[:, None],
                        np.nan)

    result: dict[int, np.ndarray] = {}
    for p in percentiles:
        frac  = p / 100.0
        bands = np.full(N_BINS, np.nan)
        for i in range(N_BINS):
            if row_counts[i] < min_count:
                continue
            idx = np.searchsorted(norm[i], frac, side="left")
            if idx < n_val_bins:
                bands[i] = bin_centres[idx]
        result[p] = bands
    return result


def smooth_band(arr: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Apply log-space Gaussian smoothing (for abundance)."""
    if sigma <= 0:
        return arr
    log_arr  = np.log10(arr)
    nan_mask = ~np.isfinite(log_arr)
    if nan_mask.all():
        return arr
    x = np.arange(N_BINS)
    filled   = np.interp(x, x[~nan_mask], log_arr[~nan_mask])
    smoothed = gaussian_filter1d(filled, sigma=sigma)
    smoothed[nan_mask] = np.nan
    result = 10.0 ** smoothed
    result  = np.where(result > 1.0, np.nan, result)
    return result


def smooth_band_phys(arr: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Log-space smoothing for physical-parameter percentile bands (no <= 1 clip)."""
    if sigma <= 0:
        return arr
    log_arr  = np.log10(arr)
    nan_mask = ~np.isfinite(log_arr)
    if nan_mask.all():
        return arr
    x = np.arange(N_BINS)
    filled   = np.interp(x, x[~nan_mask], log_arr[~nan_mask])
    smoothed = gaussian_filter1d(filled, sigma=sigma)
    smoothed[nan_mask] = np.nan
    return 10.0 ** smoothed


# ---------------------------------------------------------------------------
# 1-D model loader (unchanged)
# ---------------------------------------------------------------------------

def load_1d_data(chemistry: str):
    if chemistry == "Crich":
        inputfile = str(BASE_PATH /
                        "KoekenBak/input/20251015_Sam_Mdot_Vinf_Crich.dat")
    else:
        inputfile = str(BASE_PATH /
                        "KoekenBak/input/20251015_Sam_Mdot_Vinf_Orich.dat")
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
    return radius_1d, fracs_1d, mloss_label, vinf_label


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_single_molecule_uncertainty(
    r_grid:       np.ndarray,
    bin_sum:      np.ndarray,
    bin_count:    np.ndarray,
    hist_2d:      np.ndarray,
    molecule:     str,
    chemistry:    str,
    save_path:    Path,
    show:         bool = False,
) -> None:
    """Single-panel plot for one molecule showing 3-D mean, median, and percentiles."""
    mean = smooth_mean(bin_sum, bin_count)
    bands = compute_percentile_bands(hist_2d, AB_CENTRES)
    lo  = smooth_band(bands[PERCENTILE_LO])
    med = smooth_band(bands[50])
    hi  = smooth_band(bands[PERCENTILE_HI])

    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)
    frac_1d = get_fractional_abundance(fracs_1d, molecule)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    color = "#2166ac"

    ax.fill_between(r_grid, lo, hi, color=color, alpha=0.25,
                    label=rf"${PERCENTILE_LO}$th--${PERCENTILE_HI}$th pct.")
    ax.plot(r_grid, med, lw=2.5, ls="-", color=color, label="Median")
    ax.plot(r_grid, mean, lw=1.5, ls=":", color=color, label="Mean")

    if frac_1d is not None:
            ax.plot(radius_1d, frac_1d, lw=2, ls="-.", color=color,
                    label=f"1D ({mloss_label}, {vinf_label})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    apply_abundance_axis_limits(ax)
    add_log_ticks(ax)
    ax.set_xlabel("Radius [cm]", fontsize=14)
    ax.set_ylabel(r"Abundance relative to $\mathrm{{H}}_2$", fontsize=14)
    ax.set_title(rf"\textbf{{{molecule}}}", fontsize=14)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_phys_params(
    r_grid:      np.ndarray,
    phys_totals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path:   Path,
    show:        bool = False,
) -> None:
    """Three-panel figure for temperature, AV, density."""
    n_params = len(PHYS_PARAMS)
    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(9, 4 * n_params),
        dpi=300,
        sharex=True,
    )
    if n_params == 1:
        axes = [axes]

    for ax, (param, cfg) in zip(axes, PHYS_PARAMS.items()):
        if param not in phys_totals:
            ax.set_visible(False)
            continue

        bin_sum, bin_count, hist_2d = phys_totals[param]

        mean = smooth_mean_phys(bin_sum, bin_count)
        bands = compute_percentile_bands(
            hist_2d,
            bin_centres=cfg["centres"],
            percentiles=[PERCENTILE_LO, 50, PERCENTILE_HI],
        )
        lo  = smooth_band_phys(bands[PERCENTILE_LO])
        med = smooth_band_phys(bands[50])
        hi  = smooth_band_phys(bands[PERCENTILE_HI])

        color       = cfg["color"]
        n_particles = int(bin_count.sum())
        n_populated = int((bin_count >= MIN_COUNT).sum())

        ax.fill_between(
            r_grid, lo, hi,
            color=color, alpha=0.25,
            label=rf"${PERCENTILE_LO}$th--${PERCENTILE_HI}$th percentile",
        )
        ax.plot(r_grid, med, lw=1.5, ls=":", color=color, label="Median")
        ax.plot(r_grid, mean, lw=2.5, ls="-", color=color, label="Mean")

        ax.set_xscale("log")
        ax.set_yscale("log")
        add_log_ticks(ax)
        ax.set_ylabel(cfg["label"], fontsize=13)
        all_x, all_y = _collect_xy([ax])
        ax.set_xlim(min(all_x), max(all_x))
        ax.legend(loc="best", fontsize=11)

        print(f"  {param}: {n_populated}/{N_BINS} bins above "
              f"MIN_COUNT={MIN_COUNT}, total particles = {n_particles:,}")

    axes[-1].set_xlabel("Radius [cm]", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_avg_abundances_compare_1d(
    r_grid, averages, contributors, chemistry, save_path, show=False
):
    """Original 2x2 panel (parents/daughters, 3D vs 1D)."""
    parent_species, daughter_species, species_colors = get_species_and_colors(
        chemistry, set(averages)
    )
    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    ax_par_3d, ax_dau_3d = axes[0, 0], axes[0, 1]
    ax_par_1d, ax_dau_1d = axes[1, 0], axes[1, 1]

    for sp in parent_species:
        ax_par_3d.plot(r_grid, averages[sp], lw=3, color=species_colors[sp], label=f"{sp}")
    for sp in daughter_species:
        ax_dau_3d.plot(r_grid, averages[sp], lw=3, color=species_colors[sp], label=f"{sp}")

    ax_par_3d.set_title("Parents --- 3D median", fontsize=14)
    ax_dau_3d.set_title("Daughters --- 3D median", fontsize=14)

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

    ax_par_1d.set_title(f"Parents --- 1D ({mloss_label}, {vinf_label})", fontsize=14)
    ax_dau_1d.set_title(f"Daughters --- 1D ({mloss_label}, {vinf_label})", fontsize=14)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=12, ncol=3)
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)
        ax.set_xlabel("Radius [cm]", fontsize=14)

    for ax in axes[:, 0]:
        ax.set_ylabel("Abundance relative to $\mathrm{{H}}_2$", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_compare_1d_grid(
    r_grid,
    averages,
    contributors,
    chemistry,
    save_path,
    histograms=None,
    percentile_shading=False,
    show=False,
    n_per_panel=3,
    max_rows=3,
):
    """Per-molecule grid: solid=3-D median, dashed=1-D, optional percentile shading."""
    FIXED_COLORS = ['k', '#1f77b4', '#ff7f0e']

    parent_species, daughter_species, _ = get_species_and_colors(
        chemistry, set(averages)
    )

    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)

    all_species = parent_species + [
        s for s in daughter_species if s not in parent_species
    ]

    groups = [
        all_species[i:i + n_per_panel]
        for i in range(0, len(all_species), n_per_panel)
    ]

    if max_rows is not None and max_rows > 0:
        n_cols = min(3, len(groups))
        max_panels = max_rows * n_cols
        if len(groups) > max_panels:
            groups = groups[:max_panels]

    n_panels = len(groups)
    n_cols   = min(3, n_panels)
    n_rows   = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.5 * n_rows),
        dpi=300,
        sharex=True,
        sharey=False,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    axes_arr  = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes_arr.flatten()

    for panel_idx, group in enumerate(groups):
        ax = axes_flat[panel_idx]

        for i, sp in enumerate(group):
            color = FIXED_COLORS[i % len(FIXED_COLORS)]

            if sp in averages:
                if percentile_shading and histograms is not None:
                    bands = compute_percentile_bands(
                        histograms[sp],
                        bin_centres=AB_CENTRES,
                        percentiles=[PERCENTILE_LO, PERCENTILE_HI],
                    )
                    lo = smooth_band(bands[PERCENTILE_LO])
                    hi = smooth_band(bands[PERCENTILE_HI])
                    ax.fill_between(
                        r_grid,
                        lo,
                        hi,
                        color=color,
                        alpha=0.15,
                    )

                ax.plot(
                    r_grid,
                    averages[sp],
                    lw=3,
                    color=color,
                    ls="-",
                    label=f"{sp}",
                )

            frac_1d = get_fractional_abundance(fracs_1d, sp)
            if frac_1d is not None:
                ax.plot(
                    radius_1d,
                    frac_1d,
                    lw=3,
                    color=color,
                    ls="--",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=12, ncol=1)
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for col_idx in range(n_cols):
        for row_idx in range(n_rows - 1, -1, -1):
            if row_idx * n_cols + col_idx < n_panels:
                axes_arr[row_idx, col_idx].set_xlabel(
                    "Radius [cm]",
                    fontsize=13,
                )
                break

    for row_idx in range(n_rows):
        if row_idx * n_cols < n_panels:
            axes_arr[row_idx, 0].set_ylabel(
                r"Abundance relative to $\mathrm{{H}}_2$",
                fontsize=13,
            )

    style_handles = [
        Line2D([0], [0], color="k", lw=3, ls="-", label="3D median"),
        Line2D([0], [0], color="k", lw=3, ls="--", label=f"1D ({mloss_label}, {vinf_label})"),
    ]

    fig.legend(
        handles=style_handles,
        loc="lower center",
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fraction comparison plot (1x3 panels)
# ---------------------------------------------------------------------------

def plot_fraction_comparison(
    r_grid: np.ndarray,
    results: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    molecule: str,
    chemistry: str,
    save_path: Path,
    show: bool = False,
) -> None:
    """
    1x3 panel figure comparing abundance profiles for different particle
    subsampling fractions (10%, 50%, 100%).

    Each panel shows the shaded 16th-84th percentile band, median (solid),
    mean (dotted), and the 1-D model overlay (dashed red).
    """
    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)
    frac_1d = get_fractional_abundance(fracs_1d, molecule)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharex=True, sharey=True)

    for ax, (fraction, (bin_sum, bin_count, hist_2d)) in zip(axes, sorted(results.items())):
        mean = smooth_mean(bin_sum, bin_count)
        bands = compute_percentile_bands(
            hist_2d,
            bin_centres=AB_CENTRES,
            percentiles=[PERCENTILE_LO, 50, PERCENTILE_HI],
        )
        lo  = smooth_band(bands[PERCENTILE_LO])
        med = smooth_band(bands[50])
        hi  = smooth_band(bands[PERCENTILE_HI])

        color = "#2166ac"
        ax.fill_between(r_grid, lo, hi, color=color, alpha=0.25,
                        label=f"{PERCENTILE_LO}--{PERCENTILE_HI} pct.")
        ax.plot(r_grid, med, lw=2.5, ls="-",  color=color,     label="Median")
        ax.plot(r_grid, mean, lw=1.5, ls=":", color=color,     label="Mean")
        if frac_1d is not None:
            ax.plot(radius_1d, frac_1d, lw=2, ls="-.", color=color,
                    label=f"1D ({mloss_label}, {vinf_label})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{int(fraction * 100)}\% of particles", fontsize=13, y=1.01)
        ax.legend(loc="lower left", fontsize=10)
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)
        ax.set_xlabel("Radius [cm]", fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel(r"Abundance relative to $\mathrm{{H}}_2$", fontsize=12)

    plt.suptitle(rf"{molecule}",fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# SLURM modes
# ---------------------------------------------------------------------------

def run_plot_compare(chemistry: str, n_tasks: int, show: bool = False) -> None:
    """Aggregate partials (fraction=1.0) and produce 1D-comparison plots."""
    species_totals, _phys = aggregate_partials(chemistry, n_tasks, fraction=1.0)

    averages: dict[str, np.ndarray] = {}
    contributors: dict[str, int]    = {}

    for sp, (bin_sum, bin_count, hist_2d) in species_totals.items():
        bands = compute_percentile_bands(hist_2d, AB_CENTRES, percentiles=[50])
        averages[sp] = smooth_band(bands[50])
        contributors[sp] = int(bin_count.sum())
        n_pop = int((bin_count >= MIN_COUNT).sum())
        print(f"  {sp}: {n_pop}/{N_BINS} bins above MIN_COUNT={MIN_COUNT}, "
              f"total particles = {contributors[sp]:,}")

    save_dir = SAVE_DIR_BASE / chemistry / 'ab'
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_avg_abundances_compare_1d(
        R_CENTRES, averages, contributors, chemistry,
        save_dir / "ab_median_compare1D.png", show=show,
    )
    plot_compare_1d_grid(
        R_CENTRES, averages, contributors, chemistry,
        save_dir / "ab_median_compare1D_grid.png",
        histograms={sp: hist for sp, (_, _, hist) in species_totals.items()},
        percentile_shading=True, show=show,
    )


def run_plot_single(
    molecule: str, chemistry: str, n_tasks: int, show: bool = False,
) -> None:
    """Aggregate partials (fraction=1.0) for one molecule and plot uncertainty."""
    species_totals, _phys = aggregate_partials(chemistry, n_tasks, fraction=1.0)
    if molecule not in species_totals:
        available = sorted(species_totals.keys())
        raise KeyError(f"Molecule '{molecule}' not found.\nAvailable: {available}")

    bin_sum, bin_count, hist_2d = species_totals[molecule]
    save_dir = SAVE_DIR_BASE / chemistry / 'ab'
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_single_molecule_uncertainty(
        R_CENTRES, bin_sum, bin_count, hist_2d, molecule, chemistry,
        save_dir / f"ab_uncertainty_{molecule}.png", show=show,
    )


def run_plot_phys(chemistry: str, n_tasks: int, show: bool = False) -> None:
    """Aggregate partials (fraction=1.0) and plot physical parameters."""
    _species, phys_totals = aggregate_partials(chemistry, n_tasks, fraction=1.0)
    save_dir = SAVE_DIR_BASE / chemistry / 'ab'
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_phys_params(R_CENTRES, phys_totals, save_dir / "phys_mean.png", show=show)


def run_plot_fraction_compare(
    molecule: str, chemistry: str, n_tasks: int, show: bool = False,
) -> None:
    """Aggregate partials for fractions (10%, 50%, 100%) and create comparison plot."""
    fractions = [0.1, 0.5, 1.0]
    results = {}
    for frac in fractions:
        species_totals, _ = aggregate_partials(chemistry, n_tasks, fraction=frac)
        if molecule not in species_totals:
            available = sorted(species_totals.keys())
            raise KeyError(
                f"Molecule '{molecule}' not found in fraction {frac} data.\n"
                f"Available: {available}"
            )
        results[frac] = species_totals[molecule]
        n_pop = int((results[frac][1] >= MIN_COUNT).sum())
        print(f"Fraction {frac:.0%}: {n_pop}/{N_BINS} bins, "
              f"total particles = {int(results[frac][1].sum()):,}")

    save_dir = SAVE_DIR_BASE / chemistry / 'ab'
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_fraction_comparison(
        R_CENTRES, results, molecule, chemistry,
        save_dir / f"ab_fraction_compare_{molecule}.png", show=show,
    )


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D abundance averaging, uncertainty estimation, and 1D comparison. "
                    "Also supports particle subsampling tests."
    )
    parser.add_argument(
        "--mode",
        choices=["accumulate", "plot", "plot-single", "plot-phys", "plot-fraction-compare"],
        required=True,
        help="Mode of operation. 'plot-fraction-compare' compares 10%%, 50%%, 100%% particle subsampling.",
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
        "--molecule", type=str, default="CO",
        help="Which molecule to plot (plot-single or plot-fraction-compare mode).",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of particles to use (0.0-1.0). Only for accumulate mode.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively (plot modes only; requires a display).",
    )

    if is_interactive():
        args = parser.parse_args([
            "--mode", "plot",
            "--chemistry", "Crich",
            # "--molecule", "CH2",
            "--n-tasks", "32",
            "--show",
        ])
    else:
        args = parser.parse_args()

    if args.mode == "accumulate":
        if args.fraction < 0.0 or args.fraction > 1.0:
            raise ValueError("--fraction must be between 0 and 1")
        run_accumulate(args.task_id, args.n_tasks, args.chemistry, args.fraction)
    elif args.mode == "plot":
        run_plot_compare(args.chemistry, args.n_tasks, show=args.show)
    elif args.mode == "plot-single":
        run_plot_single(args.molecule, args.chemistry, args.n_tasks, show=args.show)
    elif args.mode == "plot-phys":
        run_plot_phys(args.chemistry, args.n_tasks, show=args.show)
    elif args.mode == "plot-fraction-compare":
        run_plot_fraction_compare(args.molecule, args.chemistry, args.n_tasks, show=args.show)


if __name__ == "__main__":
    main()