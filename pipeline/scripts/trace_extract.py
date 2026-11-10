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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_PATH
from code_chem.plotting.plot_utils import apply_abundance_axis_limits, add_log_ticks, set_plot_style, _collect_xy, format_species_label


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
# Plot mode 1: overlay traces on density slice at given dump
# ---------------------------------------------------------------------------

def run_plot(dump_index: int = 1581, xlim_au: float = 6000,
             show: bool = False) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import sarracen

    # DUMP_DIR, PHANTOM_DIR, SAVE_DIR, RAY_ANGLES_DEG and AU_PER_PC
    # are module-level globals, as in the original script.

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

    # ---- formatting (matches render_slice_v3.py / SPHRenderer style) ----
    fs = 20
    # constrained layout keeps labels, ticks and colorbars from overlapping
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.1, wspace=0.08)

    for i, (ax, plane) in enumerate(zip(axes, ("xz", "xy"))):
        y_col = "z" if plane == "xz" else "y"

        print(f"Rendering density ({plane}) at index {dump_index}...")
        sdf.render("rho", x="x", y=y_col, ax=ax,
                   xlim=(-xlim_au, xlim_au),
                   ylim=(-xlim_au, xlim_au),
                   cmap="gist_heat", log_scale=True,
                   normalize=True, cbar=False)

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

            ax.plot(x_au, y_au, color="lime", lw=2, alpha=0.9,
                    label=f"{angle}°")
            ax.plot(x_au[-1], y_au[-1], "s", color="white",
                    mec="black", mew=1.0, ms=8)

        # --- axes style: black bg, white inward ticks, black tick labels ---
        ax.set_facecolor("k")
        ax.tick_params(axis="both", which="major", direction="in",
                       length=8, width=1,
                       color="white", labelcolor="black",
                       top=True, right=True, labelsize=fs)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=4, width=1,
                       color="white",
                       top=True, right=True)
        ax.minorticks_on()
        ax.set_aspect("equal")
        ax.set_xlim(-xlim_au, xlim_au)
        ax.set_ylim(-xlim_au, xlim_au)

        ax.set_xlabel(r"$x$ [au]", fontsize=fs, color="black")
        ax.set_ylabel(rf"${y_col}$ [au]", fontsize=fs, color="black",
                      labelpad=8)
        ax.set_title(f"{plane} plane", fontsize=fs + 2, fontweight="bold",
                     color="black", pad=10)

        # --- one colorbar per panel, consistent size/gap ---
        cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log(column density)", fontsize=fs * 0.9,
                       rotation=270, labelpad=25, color="black")
        cbar.ax.tick_params(labelsize=fs * 0.8)

        if i == 0:
            ax.legend(loc="upper right", fontsize=fs * 0.8, ncol=2,
                      title="ray angle", title_fontsize=fs * 0.8,
                      framealpha=0.9)

    out = SAVE_DIR / f"trace_overlay_density_idx{dump_index:04d}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot mode 2: trace trajectories vs radius  (mirrors
#              plot_abundance_physics_comparison from evolution_traces.py)
# ---------------------------------------------------------------------------

def run_plot_traces(save_dir: Path | None = None, show: bool = False) -> None:
    """Plot every trace's trajectory (physical quantities vs radius).

    Layout mirrors plot_abundance_physics_comparison:

        4 rows x 2 columns, shared x-axis (radius, log),
        shared y-axis per row.

        Row 0 (2.2x taller): arrival time [s] vs radius [cm]
                             (i.e. how the outflow front propagates —
                             this is the "trajectory" of the trace point)
        Row 1:               number density n [cm^-3]
        Row 2:               gas temperature T [K]
        Row 3:               visual extinction A_V [mag]

        Left column:  xz plane
        Right column: xy plane

        Each panel overlays all rays from RAY_ANGLES_DEG, coloured by
        angle from +x axis.  To flip and use time as the shared x-axis,
        change `xattr = "time"` and use `d[:, 9]` for the abscissa in
        each loop — everything else works unchanged.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    # --- Load every trace file ---
    traces: dict[tuple[str, int], np.ndarray] = {}
    for plane in ("xz", "xy"):
        for angle in RAY_ANGLES_DEG:
            path = save_dir / f"trace_{plane}_{angle:03d}deg.txt"
            if path.exists():
                data = np.loadtxt(path)
                if data.ndim == 1:      # single-row guard
                    data = data[np.newaxis, :]
                traces[(plane, angle)] = data
    if not traces:
        print(f"No trace files found in {save_dir}")
        return

    angles = list(RAY_ANGLES_DEG)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(angles)))
    color_map = {a: colors[i] for i, a in enumerate(angles)}
    planes = ("xz", "xy")

    # Column indices in the trace file
    IX, IY, IZ, IDENS, ITEMP, IAV, ITIME = 0, 1, 2, 3, 4, 6, 9

    quantities = [
        (IDENS, "$n$ [cm$^{-3}$]"),
        (ITEMP, "$T$ [K]"),
        (IAV,   "$A_V$ [mag]"),
    ]

    fig, axes = plt.subplots(
        4, 2, figsize=(13, 11), dpi=300,
        sharex=True, sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1, 1, 1]},
    )
    fig.subplots_adjust(right=0.97, hspace=0.12, wspace=0.08)

    for j, plane in enumerate(planes):
        # --- Row 0: arrival time vs radius (the trajectory) ---
        ax = axes[0, j]
        for angle in angles:
            key = (plane, angle)
            if key not in traces:
                continue
            d = traces[key]
            r_cm = np.sqrt(d[:, IX]**2 + d[:, IY]**2 + d[:, IZ]**2) * CM_PER_PC
            t_s  = d[:, ITIME]
            valid = np.isfinite(r_cm) & np.isfinite(t_s) & (r_cm > 0) & (t_s > 0)
            if valid.any():
                ax.plot(r_cm[valid], t_s[valid], lw=2.5,
                        color=color_map[angle], zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"${{{plane}}}$ plane", fontsize=22, fontweight="bold", pad=12)
        if j == 0:
            ax.set_ylabel("time [s]", fontsize=18)
        else:
            ax.tick_params(labelleft=False)

        # --- Rows 1-3: physical quantities vs radius ---
        for k, (col, ylabel) in enumerate(quantities, start=1):
            axp = axes[k, j]
            for angle in angles:
                key = (plane, angle)
                if key not in traces:
                    continue
                d = traces[key]
                r_cm = np.sqrt(d[:, IX]**2 + d[:, IY]**2 + d[:, IZ]**2) * CM_PER_PC
                vals = d[:, col]
                valid = (np.isfinite(r_cm) & np.isfinite(vals)
                         & (r_cm > 0) & (vals > 0))
                if valid.any():
                    axp.plot(r_cm[valid], vals[valid], lw=2.0,
                             color=color_map[angle], zorder=3)
            axp.set_xscale("log")
            axp.set_yscale("log")
            if j == 0:
                axp.set_ylabel(ylabel, fontsize=16)
            else:
                axp.tick_params(labelleft=False)

    # --- Cosmetic formatting ---
    for i in range(4):
        for j in range(2):
            ax = axes[i, j]
            ax.grid(True, which="major", alpha=0.2, lw=0.6, zorder=0)
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.tick_params(axis="both", which="minor", labelsize=10)
            if i < 3:
                ax.tick_params(labelbottom=False)

    for j in range(2):
        axes[-1, j].set_xlabel("Radius [cm]", fontsize=18)

    # --- Legend: only include rays that actually have data ---
    handles = []
    for angle in angles:
        has_data = any(
            (plane, angle) in traces and
            np.isfinite(traces[(plane, angle)][:, ITIME]).any()
            for plane in planes
        )
        if has_data:
            handles.append(Line2D([0], [0], color=color_map[angle],
                                  lw=2.5, label=f"{angle}°"))

    if handles:
        axes[0, 1].legend(handles=handles, loc="upper left", fontsize=18,
                          frameon=True, title="Ray angle", title_fontsize=18)

    plt.tight_layout()
    out = save_dir / "trace_trajectories.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
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
    parser.add_argument("--mode",
                        choices=["extract", "plot", "plot-traces"],
                        default="extract")
    parser.add_argument("--n-workers", type=int, default=32,
                        help="Thread pool size for extract mode.")
    parser.add_argument("--dump-index", type=int, default=1581,
                        help="Dump index for plot mode.")
    parser.add_argument("--xlim-au", type=float, default=6000.0,
                        help="Half-width of plot in AU (plot mode).")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively (plot modes).")
    
    set_plot_style(dark_mode=False)

    if is_interactive():
        args = parser.parse_args([
            "--mode", "plot",
            "--show",
        ])
    else:
        args = parser.parse_args()

    if args.mode == "extract":
        run_extract(args.n_workers)
    elif args.mode == "plot":
        run_plot(args.dump_index, args.xlim_au, show=args.show)
    else:
        run_plot_traces(show=args.show)


if __name__ == "__main__":
    main()