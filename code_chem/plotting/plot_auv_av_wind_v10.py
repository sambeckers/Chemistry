"""Plot A_UV/A_V over time for all particles in traces/wind_v10."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# Set plt font to LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica"
})
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_utils import add_log_ticks

def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def _find_column(data: np.ndarray, targets: list[str]) -> np.ndarray:
    if data.dtype.names is None:
        raise ValueError("No column names found in data.")
    names = data.dtype.names
    norm_map = { _normalize_name(n): n for n in names }
    for target in targets:
        hit = norm_map.get(_normalize_name(target))
        if hit is not None:
            return data[hit]
    # Fallback: substring search
    for n in names:
        n_norm = _normalize_name(n)
        for target in targets:
            if _normalize_name(target) in n_norm:
                return data[n]
    raise KeyError(f"None of {targets} found in columns: {names}")


def _interp_to_grid(time: np.ndarray, ratio: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    mask = np.isfinite(time) & np.isfinite(ratio)
    if mask.sum() < 2:
        return None
    return np.interp(grid, time[mask], ratio[mask], left=np.nan, right=np.nan)


def load_ratios(input_dir: Path) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    files = sorted(input_dir.glob("*.phys"))
    if not files:
        raise FileNotFoundError(f"No .phys files found in {input_dir}")

    particle_ids: list[str] = []
    times: list[np.ndarray] = []
    ratios: list[np.ndarray] = []

    for file in files:
        data = np.genfromtxt(file, comments="#", names=True)
        t = np.atleast_1d(_find_column(data, ["time", "time_s", "time(s)"]))
        a_uv = np.atleast_1d(_find_column(data, ["a_uv", "auv", "a_uv_"]))
        a_v = np.atleast_1d(_find_column(data, ["a_v", "av"]))

        ratio = np.zeros_like(a_uv, dtype=float)
        valid = np.isfinite(a_uv) & np.isfinite(a_v) & (a_v > 0)
        ratio[valid] = a_uv[valid] / a_v[valid]
        ratio[~np.isfinite(ratio)] = 0.0

        particle_ids.append(file.stem)
        times.append(t.astype(float))
        ratios.append(ratio)

    return particle_ids, times, ratios


def plot_ratios(
    particle_ids: list[str],
    times: list[np.ndarray],
    ratios: list[np.ndarray],
    outpath: Path,
    time_unit: str = "yr",
) -> None:
    # Convert time to years (trace files are in seconds)
    sec_per_year = 365.25 * 24 * 3600
    time_arrays = []
    for t in times:
        if time_unit == "yr":
            time_arrays.append(t / sec_per_year)
        else:
            time_arrays.append(t)

    # Common grid for summary statistics (span all particles)
    grid = np.unique(np.concatenate(time_arrays))
    interp_stack = []
    for t, r in zip(time_arrays, ratios):
        interp = _interp_to_grid(t, r, grid)
        if interp is not None:
            interp_stack.append(interp)

    if not interp_stack:
        raise ValueError("No valid ratio data to plot.")

    interp_stack = np.vstack(interp_stack)
    median = np.nanmedian(interp_stack, axis=0)
    p16 = np.nanpercentile(interp_stack, 16, axis=0)
    p84 = np.nanpercentile(interp_stack, 84, axis=0)

    fig, ax = plt.subplots(dpi=300)

    for pid, t, r in zip(particle_ids, time_arrays, ratios):
        ax.plot(t, r, color="0.4", alpha=0.15, lw=0.8, zorder=1)

    ax.fill_between(grid, p16, p84, color="#ff9896", alpha=0.35, label="16–84\%", zorder=3)
    ax.plot(grid, median, color="#db2c2c", lw=2.0, label="Median", zorder=4)
    ax.axhline(4.65, color="#000000", lw=1.5, ls="--", label=r"$A_{UV} / A_{V}$=4.56 (Nejad+1984)", zorder=2)

    ax.set_xlabel(f"Time [{time_unit}]", fontsize=12)
    ax.set_ylabel(r"$A_{UV} / A_{V}$", fontsize=12)
    # ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    add_log_ticks(ax)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.show()

    # Console summary
    all_ratios = np.concatenate([r[np.isfinite(r)] for r in ratios])
    global_median = np.nanmedian(all_ratios)
    global_p16 = np.nanpercentile(all_ratios, 16)
    global_p84 = np.nanpercentile(all_ratios, 84)
    print(f"Global A_UV/A_V median: {global_median:.3f} (16–84%: {global_p16:.3f}–{global_p84:.3f})")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_dir = repo_root / "traces" / "wind_v10" / "trace_output_with_av"
    outpath = repo_root / "traces" / "wind_v10" / "auv_av_ratio_over_time.png"

    particle_ids, times, ratios = load_ratios(input_dir)
    plot_ratios(particle_ids, times, ratios, outpath=outpath, time_unit="yr")


if __name__ == "__main__":
    main()
