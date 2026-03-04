import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# Set plt font to LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica"
})
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
from astropy import units as u
from matplotloom import Loom

import plons
import plons.SmoothingKernelScript as sk
import plons.ConversionFactors_cgs as cgs


PC_TO_AU = (1 * u.pc).to(u.au).value
SECONDS_PER_YEAR = (1 * u.year).to(u.s).value

def _find_available_dumps(model_dir: Path) -> List[int]:
    """Find all available wind dump files in the model directory."""
    dumps = []
    for file in model_dir.glob("wind_*"):
        if file.is_file() and not file.suffix:  # wind files have no extension
            match = re.match(r"wind_(\d+)", file.name)
            if match:
                dumps.append(int(match.group(1)))
    return sorted(dumps)

def _get_dump_time_years(dump_data: plons.LoadFullDump) -> float:
    code_time = dump_data._params["time"]
    unit_time = dump_data._params["utime"]
    return (code_time * unit_time) / SECONDS_PER_YEAR

def _remove_nans(smooth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    for key in smooth.keys():
        smooth[key][np.isnan(smooth[key])] = 0.0
    return smooth

def _get_global_co_limits(traces: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    co_vals = np.concatenate([df["CO"].values for df in traces.values()])
    co_vals = co_vals[co_vals > 0]  # Filter out zeros for log scale
    return float(np.nanmin(co_vals)), float(np.nanmax(co_vals))

def _make_xz_density_slice(dump: int, model: str, lims_au: float,n_grid: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], float, plons.LoadFullDump]:
    """
    Create an x-z slice of mass density (rho) from the dump file.
    Uses all particles in the simulation (not just tracked ones).
    Returns the smooth dictionary for use with plotSlice.
    """
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir.parent / model

    setup_data = plons.LoadSetup(base_dir.parent, f"{model}/wind")
    dump_path = model_dir / f"wind_{dump:05d}"
    dump_data = plons.LoadFullDump(str(dump_path), setup_data)

    grid = np.linspace(-lims_au, lims_au, n_grid) * cgs.au
    X, Z = np.meshgrid(grid, grid)
    Y = np.zeros_like(X)

    smooth = sk.smoothMesh(X, Y, Z, dump_data, ["rho"])
    smooth = _remove_nans(smooth)

    time_years = _get_dump_time_years(dump_data)

    return X, Z, smooth, time_years, dump_data

def _load_traces(ev_dir: Path) -> Dict[str, pd.DataFrame]:
    traces: Dict[str, pd.DataFrame] = {}
    for ev_file in sorted(ev_dir.glob("ev_*.dat")):
        df = pd.read_csv(ev_file, sep=r'\s+', comment="#")
        df["X_AU"] = df["X"] * PC_TO_AU
        df["Z_AU"] = df["Z"] * PC_TO_AU
        traces[ev_file.stem] = df
    return traces

def plotSlice(ax: plt.Axes,
            X: npt.NDArray[np.single],
            Y: npt.NDArray[np.single],
            smooth: Dict[str, npt.NDArray[np.single]],
            observable: str,
            logplot: bool = False,
            fs: int = 16,
            cbar: bool = False,
            cmap: str = 'inferno',
            clim: Tuple[float, float] = (None, None)):
    """Plot a property given a grid and smoothed data ontop of the grid

    Args:
        ax (plt.Axes): axis of figure on which you want to plot the slice
        X (npt.NDArray[np.single]): X values in meshgrid which you want to plot
        Y (npt.NDArray[np.single]): Y values in meshgrid which you want to plot
        smooth (Dict[str, npt.NDArray[np.single]]): Dictionary pointing at smoothed values in meshgrid which you want to plot
        observable (str): Name of the observable you want to plot, corresponding to the name in the smooth directory
        logplot (bool, optional): plot in log scale?. Defaults to False.
        cmap (str, optional): Colormap to use. Defaults to 'inferno'.
        clim (Tuple[float, float], optional): limits for the colorbar. Defaults to (None, None).

    Returns:
        colorbar.Colorbar: Colorbar
    """

    ax.set_aspect('equal')
    ax.set_facecolor('k')

    if logplot:
        obs = np.log10(smooth[observable]+1e-99)
    else:
        obs = smooth[observable]
    axPlot = ax.pcolormesh(X/cgs.au, Y/cgs.au, obs, cmap=cmap, vmin=clim[0], vmax=clim[1], shading="nearest")
    
    if cbar == True:
        cbar = plt.colorbar(axPlot, ax=ax, location='right', fraction=0.0471, pad=0.08)  
        cbar.ax.tick_params(labelsize=fs-4)          
        return axPlot, cbar
    return axPlot, None

def _plot_trace_scatter(ax, x, z, co, norm, cmap):
    """Plot trace using scatter for smooth, high-resolution appearance"""
    if len(x) < 2:
        return ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=20, zorder=10)
    
    scatter = ax.scatter(x, z, c=co, cmap=cmap, norm=norm, s=10, zorder=10)
    return scatter

def plot_snapshot(
    model: str,
    dump: int,
    ev_dir: Path,
    lims_au: float = 5000.0,
    n_grid: int = 600,
    cmap_rho: str = "inferno",
    cmap_co: str = "viridis",
    dark_mode: bool = False,
) -> plt.Figure:
    """Plot snapshot at a specific dump showing density background and trace evolution up to that time.
    
    Parameters
    ----------
    dark_mode : bool, default=False
        If True, use black background with white text for publication-quality figures
    """
    traces = _load_traces(ev_dir)
    vmin_co, vmax_co = _get_global_co_limits(traces)
    norm_co = LogNorm(vmin=vmin_co, vmax=vmax_co)

    X, Z, smooth, time_years, dump_data = _make_xz_density_slice(
        dump=dump,
        model=model,
        lims_au=lims_au,
        n_grid=n_grid,
    )

    # Calculate colormap limits using min/max approach from column_densities.py
    vmin_rho = np.log10(np.min(np.array(dump_data['rho'][np.array(dump_data['rho']) > 0])))
    vmax_rho = np.log10(np.amax(smooth['rho']))

    # Set up styling based on dark_mode
    rc_context = {
        'axes.edgecolor': 'white' if dark_mode else 'black',
        'xtick.color': 'white' if dark_mode else 'black',
        'ytick.color': 'white' if dark_mode else 'black',
        'figure.facecolor': 'black' if dark_mode else 'white',
        'axes.facecolor': 'black' if dark_mode else 'white',
        'text.color': 'white' if dark_mode else 'black',
    }
    label_color = 'white' if dark_mode else 'black'

    with plt.rc_context(rc_context):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        _, cbar_rho = plotSlice(ax, X, Z, smooth, 'rho', logplot=True, cmap=cmap_rho, cbar=True,
                             clim=(vmin_rho, vmax_rho), fs=12)
        cbar_rho.set_label(r"log $\rho$ [g/cm$^3$]", fontsize=12, color=label_color)
        if dark_mode:
            cbar_rho.ax.yaxis.set_tick_params(color=label_color)
            plt.setp(plt.getp(cbar_rho.ax.axes, 'yticklabels'), color=label_color)

        # Plot traces only up to current time (like in animation)
        for df in traces.values():
            mask = df["TIME"].values <= time_years
            if mask.any():
                x = df.loc[mask, "X_AU"].values
                z = df.loc[mask, "Z_AU"].values
                co = df.loc[mask, "CO"].values
                _plot_trace_scatter(ax, x, z, co, norm_co, cmap_co)

        # Create separate axes for CO colorbar to avoid interfering with rho colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax_co = divider.append_axes("right", size="5%", pad=0.1)
        if dark_mode:
            cax_co.set_facecolor('black')
        
        sm = plt.cm.ScalarMappable(norm=norm_co, cmap=cmap_co)
        sm.set_array([])
        cbar_co = fig.colorbar(sm, cax=cax_co)
        cbar_co.set_label("CO abundance", fontsize=12, color=label_color)
        if dark_mode:
            cbar_co.ax.yaxis.set_tick_params(color=label_color)
            plt.setp(plt.getp(cbar_co.ax.axes, 'yticklabels'), color=label_color)

        ax.set_xlabel("x [au]", fontsize=12, color=label_color)
        ax.set_ylabel("z [au]", fontsize=12, color=label_color)
        ax.set_title(f"{model} | dump {dump} | t={time_years:.2f} yr", fontsize=12, color=label_color)

        return fig

def main() -> None:
    model = "v10_a09"
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir.parent / model
    ev_dir = model_dir / "ev_output"

    dump_set = 1600
    lims_au = 5000.0
    n_grid = 600

    dumps = _find_available_dumps(model_dir)
    dumps = [d for d in dumps]
    dump_final = dumps[-1]

    if not dumps:
        raise RuntimeError("No dump files found in model directory.")

    traces = _load_traces(ev_dir)
    trace_min = min(df["TIME"].min() for df in traces.values())
    trace_max = max(df["TIME"].max() for df in traces.values())

    print(f"Trace time range: {trace_min:.2f} - {trace_max:.2f} yr")

    first_dump_time = None
    last_dump_time = None
    for dump in tqdm((dumps[0], dumps[-1]), desc="Calculating dump times"):
        _, _, _, t_years, _ = _make_xz_density_slice(
            dump=dump,
            model=model,
            lims_au=lims_au,
            n_grid=n_grid,
        )
        if first_dump_time is None:
            first_dump_time = t_years
        else:
            last_dump_time = t_years

    if first_dump_time is not None and last_dump_time is not None:
        print(f"Dump time range: {first_dump_time:.2f} - {last_dump_time:.2f} yr")
        if last_dump_time < trace_min or first_dump_time > trace_max:
            print("WARNING: Dump times do not overlap trace times.")

    fig = plot_snapshot(
                model=model,
                dump=dump_set,
                ev_dir=ev_dir,
                lims_au=lims_au,
                n_grid=n_grid,
                dark_mode=True,
            )
    fig.savefig(base_dir.parent / 'figures' / model / f"{dump_set}_dens_trace_snapshot.png", dpi=300)

    n_frames = 100
    idx = np.linspace(0, len(dumps) - 1, n_frames, dtype=int)
    frames = [dumps[i] for i in idx]
    print(f"Selected frames for snapshot: {frames}")

    loom = Loom(
    str(base_dir.parent / 'figures'/ model / f"dens_trace.mp4"),
    frames_directory= str(base_dir.parent / 'figures' / model / 'frames'),
    fps = 30,
    overwrite = True,
    verbose = True,
    savefig_kwargs = {
        "dpi": 300,
    }
    )
    # with loom:
    #     for dump in tqdm(frames, desc="Creating snapshot frames"):
    #         fig = plot_snapshot(
    #             model=model,
    #             dump=dump,
    #             ev_dir=ev_dir,
    #             lims_au=lims_au,
    #             n_grid=n_grid,
    #         )
    #         loom.save_frame(fig)
    #         plt.close(fig)
        

if __name__ == "__main__":
    main()
