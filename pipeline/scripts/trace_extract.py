#!/usr/bin/env python3
"""
trace_extract.py

Extract particle-property traces along specific ray directions in the xy
and xz planes, from a sequence of Phantom HDF5 dumps.

Output files mimic particle_slice_0.txt — one line per timestep:

    XCOORD(pc) YCOORD(pc) ZCOORD(pc) DENS(cm-3) TEMPGAS(K) TEMPDUST(K) AV(mag) ZETAXR ZETACR TIME(s)

At each timestep, the (x, y, z) shown is the "trace point" along the ray —
the radius where the outflow front is at that time — so it moves outward
over the simulation (like a virtual particle).

HDF5 units assumed (per user clarification):
    x, y, z : cm
    density : cm^-3
    temp    : K
    av      : mag
    time    : s

No SLURM accumulation is needed: total output is ~2 MB.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUMP_DIR    = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
PHANTOM_DIR = Path("/fred/oz304/beckers/v10a09")
SAVE_DIR    = Path("/fred/oz304/beckers/Chemistry/traces/v10a09_out")

# Radial grid (cm)
R_MIN, R_MAX, N_BINS = 1e14, 1e18, 100
R_EDGES   = np.logspace(np.log10(R_MIN), np.log10(R_MAX), N_BINS + 1)
R_CENTRES = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])

# Ray directions from +x axis (degrees), first quadrant only
RAY_ANGLES_DEG     = [0, 15, 30, 45, 60, 75, 90]
RAY_HALF_WIDTH_DEG = 5.0

# Trace point definition
#   "outer_mass" : radius where outer cumulative mass < MASS_OUTER_FRACTION
#   "peak_density": radius of maximum bin density
TRACE_METHOD         = "outer_mass"
MASS_OUTER_FRACTION  = 0.10

# Unit conversions
CM_PER_PC = 3.0857e18
AU_PER_PC = 206264.806


# ---------------------------------------------------------------------------
# Trace point selection
# ---------------------------------------------------------------------------

def _trace_point(r_sel, dens_sel, temp_sel, av_sel, t_s):
    """Pick one representative (r, dens, temp, av, t) for the ray at this dump."""
    idx   = np.digitize(r_sel, R_EDGES) - 1
    valid = (idx >= 0) & (idx < N_BINS)
    if not valid.any():
        return None
    idx_v = idx[valid]

    bin_count = np.bincount(idx_v, minlength=N_BINS).astype(np.int64)
    populated = bin_count > 0
    if not populated.any():
        return None

    n_safe = np.maximum(bin_count, 1).astype(np.float64)
    bin_dens = np.bincount(idx_v, weights=dens_sel[valid], minlength=N_BINS) / n_safe
    bin_temp = np.bincount(idx_v, weights=temp_sel[valid], minlength=N_BINS) / n_safe
    bin_av   = np.bincount(idx_v, weights=av_sel[valid],   minlength=N_BINS) / n_safe
    bin_dens[~populated] = 0.0

    if TRACE_METHOD == "peak_density":
        r_idx = int(np.argmax(bin_dens))
    else:  # "outer_mass"
        dr = np.diff(R_EDGES)
        bin_mass = bin_dens * R_CENTRES**2 * dr
        total_mass = bin_mass.sum()
        if total_mass <= 0:
            return None
        # Cumulative mass from the OUTSIDE inward
        cumsum_outside = np.cumsum(bin_mass[::-1])[::-1]
        target = MASS_OUTER_FRACTION * total_mass
        below = np.where(cumsum_outside < target)[0]
        r_idx = int(below[0]) if below.size else N_BINS - 1
        # Step inward until we hit a populated bin
        while r_idx > 0 and not populated[r_idx]:
            r_idx -= 1

    return {
        "r_cm": R_CENTRES[r_idx],
        "dens": float(bin_dens[r_idx]),
        "temp": float(bin_temp[r_idx]),
        "av":   float(bin_av[r_idx]),
        "time": float(t_s),
    }


# ---------------------------------------------------------------------------
# Per-dump worker
# ---------------------------------------------------------------------------

def _process_dump(path: Path):
    """Return dict[(plane, angle_deg)] -> state | None for one dump."""
    with h5py.File(path, "r") as f:
        x    = f["particles/x"][:].astype(np.float64)
        y    = f["particles/y"][:].astype(np.float64)
        z    = f["particles/z"][:].astype(np.float64)
        dens = f["particles/density"][:].astype(np.float64)
        temp = f["particles/temp"][:].astype(np.float64)
        av   = f["particles/av"][:].astype(np.float64)

        t_s = None
        for loc in ("time", "particles/time", "header/time"):
            try:
                t_s = float(np.asarray(f[loc]).flat[0])
                break
            except KeyError:
                continue
        if t_s is None:
            t_s = 0.0

    base = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
        np.isfinite(dens) & np.isfinite(temp) & np.isfinite(av) &
        (dens > 0) & (temp > 0) & (av >= 0)
    )

    result = {}
    for plane in ("xz", "xy"):
        if plane == "xz":
            r   = np.sqrt(x * x + z * z)
            phi = np.arctan2(z, x)
        else:
            r   = np.sqrt(x * x + y * y)
            phi = np.arctan2(y, x)
        base_plane = base & np.isfinite(r) & (r > 0)

        for theta_deg in RAY_ANGLES_DEG:
            theta  = np.radians(theta_deg)
            dtheta = np.radians(RAY_HALF_WIDTH_DEG)

            # First-quadrant wedge around the ray
            angle_mask = (
                (phi >= 0.0) &
                (phi <= np.pi / 2.0) &
                (np.abs(phi - theta) < dtheta)
            )
            mask = base_plane & angle_mask
            if not mask.any():
                result[(plane, theta_deg)] = None
                continue

            result[(plane, theta_deg)] = _trace_point(
                r[mask], dens[mask], temp[mask], av[mask], t_s
            )
    return result


def _safe_process_dump(path: Path):
    try:
        return _process_dump(path)
    except Exception as exc:
        print(f"  Warning: skipping {path.name}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_trace_file(path: Path, plane: str, angle_deg: int,
                     states: list[dict | None]) -> None:
    """Write one trace file with `len(states)` rows (one per timestep)."""
    theta = np.radians(angle_deg)
    rows = []
    for state in states:
        if state is None:
            rows.append([np.nan] * 10)
            continue
        r_cm = state["r_cm"]
        # Analytic ray coordinates (matches sample: perpendicular = 0)
        if plane == "xz":
            x_pc = r_cm * np.cos(theta) / CM_PER_PC
            y_pc = 0.0
            z_pc = r_cm * np.sin(theta) / CM_PER_PC
        else:
            x_pc = r_cm * np.cos(theta) / CM_PER_PC
            y_pc = r_cm * np.sin(theta) / CM_PER_PC
            z_pc = 0.0
        rows.append([
            x_pc, y_pc, z_pc,
            state["dens"],
            state["temp"], state["temp"],   # TEMPGAS == TEMPDUST
            state["av"],
            0.0, 0.0,                       # ZETAXR == ZETACR == 0
            state["time"],
        ])

    np.savetxt(path, np.array(rows), fmt="%.17e", delimiter="  ")


# ---------------------------------------------------------------------------
# Extract mode
# ---------------------------------------------------------------------------

def run_extract(n_workers: int = 32) -> None:
    dump_files = sorted(DUMP_DIR.glob("dump_*.h5"))
    if not dump_files:
        raise FileNotFoundError(f"No dumps in {DUMP_DIR}")
    print(f"Processing {len(dump_files)} dumps with {n_workers} workers")

    # Pool.map preserves input order → results[i] corresponds to dump_files[i]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        all_results = list(tqdm(
            pool.map(_safe_process_dump, dump_files),
            total=len(dump_files), desc="Dumps",
        ))

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_rows  = 0
    for plane in ("xz", "xy"):
        for angle_deg in RAY_ANGLES_DEG:
            states = [r.get((plane, angle_deg)) for r in all_results]
            filename = SAVE_DIR / f"trace_{plane}_{angle_deg:03d}deg.txt"
            _write_trace_file(filename, plane, angle_deg, states)
            n_valid = sum(s is not None for s in states)
            print(f"  {filename.name}: {len(states)} rows "
                  f"({n_valid} valid)")
            n_files += 1
            n_rows += len(states)

    print(f"\nWrote {n_files} files ({n_rows} rows total) → {SAVE_DIR}")


# ---------------------------------------------------------------------------
# Plot mode: overlay traces on density slice at given dump
# ---------------------------------------------------------------------------

def run_plot(dump_index: int = 1581, xlim_au: float = 6000,
             show: bool = False) -> None:
    import matplotlib.pyplot as plt
    import sarracen

    dump_files = sorted(DUMP_DIR.glob("dump_*.h5"))
    if dump_index >= len(dump_files):
        print(f"dump_index {dump_index} out of range (0-{len(dump_files)-1})")
        return
    dump_path = dump_files[dump_index]
    stem = dump_path.stem.split("_", 1)[1]
    phantom_path = PHANTOM_DIR / f"wind_{stem}"
    if not phantom_path.exists():
        print(f"Phantom file not found: {phantom_path}")
        return

    traces = {}
    for plane in ("xz", "xy"):
        for angle in RAY_ANGLES_DEG:
            path = SAVE_DIR / f"trace_{plane}_{angle:03d}deg.txt"
            if path.exists():
                traces[(plane, angle)] = np.loadtxt(path)
    if not traces:
        print(f"No trace files found in {SAVE_DIR}")
        return

    print(f"Loading phantom: {phantom_path.name}")
    sdf, _ = sarracen.read_phantom(phantom_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200)

    for ax, plane in zip(axes, ("xz", "xy")):
        y_col = "z" if plane == "xz" else "y"

        print(f"Rendering density ({plane}) at index {dump_index}...")
        sdf.render("rho", x="x", y=y_col, ax=ax,
                   xlim=(-xlim_au, xlim_au),
                   ylim=(-xlim_au, xlim_au),
                   cmap="gist_heat", log_scale=True,
                   normalize=True, dens_weight=True)

        for angle in RAY_ANGLES_DEG:
            key = (plane, angle)
            if key not in traces:
                continue
            data = traces[key]
            x_pc = data[:, 0]
            y_pc = data[:, 1]
            z_pc = data[:, 2]

            if plane == "xz":
                x_au = x_pc * AU_PER_PC
                y_au = z_pc * AU_PER_PC
            else:
                x_au = x_pc * AU_PER_PC
                y_au = y_pc * AU_PER_PC

            ax.plot(x_au, y_au, color="lime", lw=1.5, alpha=0.85,
                    label=f"{angle}°")
            # Final-timestep marker
            ax.plot(x_au[-1], y_au[-1], "s", color="white",
                    mec="black", mew=0.8, ms=5)

        ax.set_title(f"{plane} plane  (t = {dump_path.name})", fontsize=13)
        ax.set_xlabel("x [au]", fontsize=12)
        ax.set_ylabel(f"{y_col} [au]", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(loc="upper right", fontsize=9, ncol=2,
                  title="ray angle", title_fontsize=9)

    plt.tight_layout()
    out = SAVE_DIR / f"trace_overlay_density_idx{dump_index:04d}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def is_interactive():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["extract", "plot"], default="extract")
    parser.add_argument("--n-workers", type=int, default=32,
                        help="Thread pool size for extract mode.")
    parser.add_argument("--dump-index", type=int, default=1581,
                        help="Dump index for plot mode.")
    parser.add_argument("--xlim-au", type=float, default=6000.0,
                        help="Half-width of plot in AU (plot mode).")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively (plot mode).")

    if is_interactive():
        args = parser.parse_args([
            "--mode", "plot",
            "--show",
        ])
    else:
        args = parser.parse_args()
    
    if args.mode == "extract":
        run_extract(args.n_workers)
    else:
        run_plot(args.dump_index, args.xlim_au, show=args.show)


if __name__ == "__main__":
    main()