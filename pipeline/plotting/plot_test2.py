"""
plot_mean_abundance_vs_radius.py

Computes the mean abundance of one or more species as a function of radius
by accumulating ALL particles from ALL dump files into log-spaced radius bins.

Why this is smoother than per-dump averaging
--------------------------------------------
Each dump is a snapshot: particles cluster at their current radii, leaving
most radius bins empty or poorly sampled.  By accumulating across all 1600
dumps every bin receives contributions from thousands of particles, giving
the same statistical quality as trajectory-based averaging — without needing
to reconstruct individual trajectories.

A final Gaussian smooth in log-radius space (matching the effective smoothing
from trajectory interpolation in the existing ev_*.dat pipeline) removes
residual shot noise in sparsely-populated bins at the outer envelope.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica",
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUMP_DIR  = Path("/aphid/scratch-3month/sbeckers/v10a09_out/output/dumps")
SAVE_PATH = Path("ab_mean_vs_radius_not_smooth.pdf")

SPECIES   = ["co"]          # extend to plot multiple species, e.g. ["co", "h2o"]
LABELS    = {"co": "CO"}    # display label per species

# Radius grid (cm) — adjust to match your simulation domain
R_MIN  = 1e13
R_MAX  = 1e17
N_BINS = 300                # more bins = finer resolution before smoothing

# Gaussian smooth width in number of bins (log-spaced).
# sigma=3 gives ~similar smoothing to trajectory interpolation + nanmean.
# Increase if still noisy, decrease to preserve sharp features.
SMOOTH_SIGMA = 3

# Minimum particle count per bin to trust the mean (bins below this → NaN)
MIN_COUNT = 30

MAX_WORKERS = 128

# ---------------------------------------------------------------------------
# Radius bin edges (log-spaced, computed once)
# ---------------------------------------------------------------------------

R_EDGES   = np.logspace(np.log10(R_MIN), np.log10(R_MAX), N_BINS + 1)
R_CENTRES = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])   # geometric centre of each bin


# ---------------------------------------------------------------------------
# Per-dump worker: returns compact bin-sum and bin-count arrays
# ---------------------------------------------------------------------------

def _accumulate_dump(path: Path, species_list: list[str]):
    """
    Read r and each species array from one dump file.
    Returns {species: (bin_sum, bin_count)} — only N_BINS floats each,
    so memory per worker is negligible regardless of particle count.
    """
    import h5py  # import inside worker for thread safety with some h5py builds

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    try:
        with h5py.File(path, "r") as f:
            r = f["particles/r"][:]

            for species in species_list:
                try:
                    ab = f[f"particles/{species}"][:]
                except KeyError:
                    continue

                mask = (
                    np.isfinite(r)  &
                    np.isfinite(ab) &
                    (r  > 0)        &
                    (ab > 0)
                )
                r_ok  = r[mask]
                ab_ok = ab[mask]

                if r_ok.size == 0:
                    continue

                idx   = np.digitize(r_ok, R_EDGES) - 1
                valid = (idx >= 0) & (idx < N_BINS)

                bin_sum = np.bincount(idx[valid], weights=ab_ok[valid], minlength=N_BINS)
                bin_cnt = np.bincount(idx[valid],                        minlength=N_BINS)

                result[species] = (bin_sum, bin_cnt.astype(np.int64))

    except Exception as exc:
        print(f"  Warning: failed to read {path.name}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Main accumulation loop
# ---------------------------------------------------------------------------

def accumulate_all_dumps(
    dump_files: list[Path],
    species_list: list[str],
    max_workers: int = MAX_WORKERS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {species: (total_sum array, total_count array)} across all dumps.
    """
    totals: dict[str, list] = {s: [np.zeros(N_BINS), np.zeros(N_BINS, dtype=np.int64)]
                                for s in species_list}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_accumulate_dump, p, species_list): p
                   for p in dump_files}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Accumulating dumps"):
            partial = fut.result()
            for species, (s, c) in partial.items():
                totals[species][0] += s
                totals[species][1] += c

    return {sp: (totals[sp][0], totals[sp][1]) for sp in species_list}


# ---------------------------------------------------------------------------
# Smooth mean in log-space
# ---------------------------------------------------------------------------

def smooth_mean(
    bin_sum: np.ndarray,
    bin_count: np.ndarray,
    min_count: int = MIN_COUNT,
    sigma: float = SMOOTH_SIGMA,
) -> np.ndarray:
    """
    Compute per-bin mean, optionally apply Gaussian smoothing in log space.

    Behaviour:
    ----------
    sigma <= 0:
        No smoothing; return the raw mean profile.

    sigma > 0:
        Smooth in log-abundance space after interpolating across NaN gaps.
        This mimics the effective smoothing from trajectory interpolation
        used in the ev_*.dat pipeline.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(bin_count >= min_count,
                        bin_sum / bin_count,
                        np.nan)

    # No smoothing requested
    if sigma <= 0:
        return mean

    # Convert to log space for multiplicative smoothing behaviour
    log_mean = np.log10(mean)

    # Identify invalid bins
    nan_mask = ~np.isfinite(log_mean)

    # If everything is invalid, nothing to smooth
    if nan_mask.all():
        return mean

    # Fill NaN gaps before Gaussian filtering
    x = np.arange(N_BINS)
    log_filled = np.interp(
        x,
        x[~nan_mask],
        log_mean[~nan_mask],
    )

    # Gaussian smooth in log-radius space
    log_smoothed = gaussian_filter1d(log_filled, sigma=sigma)

    # Restore original invalid regions
    log_smoothed[nan_mask] = np.nan

    return 10.0 ** log_smoothed


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(
    r_centres: np.ndarray,
    means: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    save_path: Path,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for species, mean in means.items():
        n_total = int(counts[species].sum())
        label   = LABELS.get(species, species)
        ax.plot(r_centres, mean, lw=2, label=rf"{label}  (N$_\mathrm{{particles}}$={n_total:,})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius [cm]", fontsize=14)
    ax.set_ylabel(r"Abundance (wrt H$_\mathrm{nuc}$)", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dump_files = sorted(DUMP_DIR.glob("dump_*.h5"))
    if not dump_files:
        raise FileNotFoundError(f"No dump_*.h5 files found in {DUMP_DIR}")
    print(f"Found {len(dump_files)} dump files")

    totals = accumulate_all_dumps(dump_files, SPECIES)

    means  = {}
    counts = {}
    for species, (bin_sum, bin_count) in totals.items():
        means[species]  = smooth_mean(bin_sum, bin_count)
        counts[species] = bin_count
        n_populated = int((bin_count >= MIN_COUNT).sum())
        print(f"  {species}: {n_populated}/{N_BINS} bins above MIN_COUNT={MIN_COUNT}")

    plot(R_CENTRES, means, counts, SAVE_PATH, show=True)


if __name__ == "__main__":
    main()