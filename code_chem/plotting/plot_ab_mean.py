import math
import os
import re
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica",
})

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_PATH, daughters_Crich, daughters_Orich, parents
from n_distinct_colours import generate_colormap
from kb.modeling.tools import CodeIO
from kb import path as kb_path
from run_plot_single_config_1D_model import (
    run_model,
    find_star_for_config,
    get_fractional_abundance,
    MLOSS,
    VELOCITY,
    TARGET_MLOSS,
    TARGET_VELOCITY,
)
from plot_utils import apply_abundance_axis_limits, apply_abundance_axis_limits_shared, add_log_ticks


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else BASE_PATH / path


def particle_id_from_file(path):
    match = re.match(r"ev_(\d+)\.dat$", path.name)
    return int(match.group(1)) if match else -1


def list_particle_files(source_dir):
    files = sorted(source_dir.glob("ev_*.dat"), key=particle_id_from_file)
    return [f for f in files if particle_id_from_file(f) >= 0]


def files_from_particle_ids(source_dir, particle_ids):
    files = []
    for pid in particle_ids:
        file_path = source_dir / f"ev_{pid}.dat"
        if file_path.exists():
            files.append(file_path)
    return files


def load_particle_data(files, min_rows=10):
    all_data = []
    for filepath in files:
        data = np.genfromtxt(filepath, comments="#", skip_header=4, names=True)
        if data.ndim == 0:
            data = np.array([data])
        if len(data) < min_rows:
            continue
        all_data.append(data)
    return all_data


def common_radius_grid(all_data, n_points):
    rmins = []
    rmaxs = []
    for data in all_data:
        r = data["R"]
        mask = np.isfinite(r) & (r > 0)
        if not np.any(mask):
            continue
        rmins.append(np.min(r[mask]))
        rmaxs.append(np.max(r[mask]))

    if not rmins or not rmaxs:
        raise ValueError("No valid positive radius values were found in the input files.")

    # Use the union of all radius ranges so that every particle contributes
    # where it has data; points outside a particle's range are NaN (handled
    # by np.nanmean in average_species).
    rmin = min(rmins)
    rmax = max(rmaxs)

    return np.logspace(np.log10(rmin), np.log10(rmax), n_points)


def interpolate_species_on_grid(data, species, r_grid):
    if species not in data.dtype.names:
        return None

    r = data["R"]
    y = data[species]
    mask = np.isfinite(r) & np.isfinite(y) & (r > 0) & (y >= 0)
    if np.sum(mask) < 2:
        return None

    r_valid = r[mask]
    y_valid = y[mask]

    order = np.argsort(r_valid)
    r_sorted = r_valid[order]
    y_sorted = y_valid[order]

    unique_r, unique_idx = np.unique(r_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    if len(unique_r) < 2:
        return None

    y_interp = np.interp(
        np.log10(r_grid),
        np.log10(unique_r),
        y_unique,
        left=np.nan,
        right=np.nan,
    )
    return y_interp


def average_species(all_data, species_list, r_grid):
    averages = {}
    contributors = {}

    for species in species_list:
        curves = []
        for data in all_data:
            curve = interpolate_species_on_grid(data, species, r_grid)
            if curve is not None:
                curves.append(curve)

        if not curves:
            continue

        stack = np.vstack(curves)
        averages[species] = np.nanmean(stack, axis=0)
        contributors[species] = stack.shape[0]

    return averages, contributors


def _get_species_and_colors(averages, chemistry):
    daughters = daughters_Orich if chemistry == "Orich" else daughters_Crich

    parent_species = [s for s in parents if s in averages]
    daughter_species = [s for s in daughters if s in averages]

    if not parent_species and not daughter_species:
        raise ValueError("None of the parent/daughter species were found in loaded data.")

    all_species = parent_species + [s for s in daughter_species if s not in parent_species]
    cmap = generate_colormap(len(all_species)).colors
    species_colors = {species: cmap[i] for i, species in enumerate(all_species)}

    return parent_species, daughter_species, species_colors

def plot_avg_abundances(r_grid, averages, contributors, chemistry, save_path, show=False):
    parent_species, daughter_species, species_colors = _get_species_and_colors(averages, chemistry)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharex=True, sharey=False)

    for species in parent_species:
        axes[0].plot(
            r_grid,
            averages[species],
            lw=2,
            color=species_colors[species],
            label=f"{species} (N={contributors[species]})",
        )

    for species in daughter_species:
        axes[1].plot(
            r_grid,
            averages[species],
            lw=2,
            color=species_colors[species],
            label=f"{species} (N={contributors[species]})",
        )

    axes[0].set_title("Parent molecules", fontsize=16)
    axes[1].set_title("Daughter molecules", fontsize=16)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9, ncol=3)
        ax.set_xlabel("Radius [cm]", fontsize=14)
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)

    axes[0].set_ylabel("Abundance (wrt H$_{nuc}$)", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

def load_1d_data(chemistry):
    """Load 1-D model radius and fractional abundances for the configured target."""
    if chemistry == "Crich":
        inputfile = str(BASE_PATH / "KoekenBak/input/20251015_Sam_Mdot_Vinf_Crich.dat")
    else:
        inputfile = str(BASE_PATH / "KoekenBak/input/20251015_Sam_Mdot_Vinf_Orich.dat")

    model = run_model(inputfile)
    idx, star, mloss_label, vinf_label = find_star_for_config(
        model, TARGET_MLOSS, TARGET_VELOCITY
    )
    folder = os.path.join(kb_path.cout, "models", star["LAST_CHEMISTRY_MODEL"]) + "/"
    radius_1d = CodeIO.getChemistryPhysPar(folder + "csphyspar_smooth.out", "RADIUS")
    fracs_1d = CodeIO.getChemistryAbundances(folder + "csfrac_smooth.out")
    return radius_1d, fracs_1d, mloss_label, vinf_label

def plot_avg_abundances_compare_1d(
    r_grid, averages, contributors, chemistry, save_path, show=False
):
    """Plot 3-D averaged abundances (top row) and 1-D model abundances (bottom row).

    The figure has a 2 x 2 layout:
        Row 0 – 3-D particle-averaged parents  |  daughters
        Row 1 – 1-D model parents              |  daughters
    """
    parent_species, daughter_species, species_colors = _get_species_and_colors(
        averages, chemistry
    )

    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    ax_par_3d, ax_dau_3d = axes[0, 0], axes[0, 1]
    ax_par_1d, ax_dau_1d = axes[1, 0], axes[1, 1]

    # ── Top row: 3-D averages ──────────────────────────────────────────────────
    for species in parent_species:
        ax_par_3d.plot(
            r_grid,
            averages[species],
            lw=3,
            color=species_colors[species],
            label=f"{species} (N={contributors[species]})",
        )

    for species in daughter_species:
        ax_dau_3d.plot(
            r_grid,
            averages[species],
            lw=3,
            color=species_colors[species],
            label=f"{species} (N={contributors[species]})",
        )

    ax_par_3d.set_title("Parents — 3D average", fontsize=14)
    ax_dau_3d.set_title("Daughters — 3D average", fontsize=14)

    # ── Bottom row: 1-D model ─────────────────────────────────────────────────
    for species in parent_species:
        frac = get_fractional_abundance(fracs_1d, species)
        if frac is None:
            print(f"1-D: skipping missing parent: {species}")
            continue
        ax_par_1d.plot(
            radius_1d,
            frac,
            lw=3,
            color=species_colors[species],
            label=species,
        )

    for species in daughter_species:
        frac = get_fractional_abundance(fracs_1d, species)
        if frac is None:
            print(f"1-D: skipping missing daughter: {species}")
            continue
        ax_dau_1d.plot(
            radius_1d,
            frac,
            lw=3,
            color=species_colors[species],
            label=species,
        )

    ax_par_1d.set_title(
        f"Parents — 1D ({mloss_label}, {vinf_label})", fontsize=14
    )
    ax_dau_1d.set_title(
        f"Daughters — 1D ({mloss_label}, {vinf_label})", fontsize=14
    )

    # ── Shared formatting ─────────────────────────────────────────────────────
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9, ncol=3)
    # Apply limits per-axis (no sharey/sharex across rows and columns)
    for ax in axes.flat:
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)

    for ax in axes[1, :]:
        ax.set_xlabel("Radius [cm]", fontsize=14)
    for ax in axes[0, :]:
        ax.set_xlabel("Radius [cm]", fontsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel("Abundance (wrt H$_{nuc}$)", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_compare_1d_grid(
    r_grid, averages, contributors, chemistry, save_path, show=False,
    n_per_panel=3,
):
    """Thorough per-molecule comparison: 3-D average (solid) vs 1-D model (dashed).

    Molecules are grouped into panels of at most *n_per_panel* (default 3),
    laid out in a grid of up to 4 columns.  All panels share the x-axis;
    y-limits are set independently per panel based on the highest-abundance
    molecule in that panel (10 decades of dynamic range).

    A global legend at the bottom of the figure shows the solid/dashed
    convention for 3-D vs 1-D.
    """
    parent_species, daughter_species, species_colors = _get_species_and_colors(
        averages, chemistry
    )
    radius_1d, fracs_1d, mloss_label, vinf_label = load_1d_data(chemistry)

    # Ordered list: parents first, then daughters
    all_species = parent_species + [s for s in daughter_species if s not in parent_species]

    # Group into panels of n_per_panel
    groups = [all_species[i:i + n_per_panel]
              for i in range(0, len(all_species), n_per_panel)]
    n_panels = len(groups)
    n_cols = min(4, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.2 * n_cols, 4.5 * n_rows),
        dpi=300,
        sharex=True,
        sharey=False,
    )
    axes_arr = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes_arr.flatten()

    for panel_idx, group in enumerate(groups):
        ax = axes_flat[panel_idx]

        for species in group:
            color = species_colors[species]

            # 3-D average — solid line
            if species in averages:
                n = contributors.get(species, 0)
                ax.plot(
                    r_grid,
                    averages[species],
                    lw=3, color=color, ls='-',
                    label=f"{species} (N={n})",
                )

            # 1-D model — dashed line, same colour
            frac_1d = get_fractional_abundance(fracs_1d, species)
            if frac_1d is not None:
                ax.plot(
                    radius_1d, frac_1d,
                    lw=3, color=color, ls='--',
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9, ncol=1)
        apply_abundance_axis_limits(ax)
        add_log_ticks(ax)

    # Hide unused panels in the last row
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # x-axis label: last visible panel in each column
    for col_idx in range(n_cols):
        for row_idx in range(n_rows - 1, -1, -1):
            if row_idx * n_cols + col_idx < n_panels:
                axes_arr[row_idx, col_idx].set_xlabel("Radius [cm]", fontsize=13)
                break

    # y-axis label: left column only
    for row_idx in range(n_rows):
        if row_idx * n_cols < n_panels:
            axes_arr[row_idx, 0].set_ylabel(
                "Abundance (wrt H$_{\\mathrm{nuc}}$)", fontsize=13
            )

    # Global legend: solid = 3-D, dashed = 1-D
    style_handles = [
        Line2D([0], [0], color='k', lw=3, ls='-',  label="3D average"),
        Line2D([0], [0], color='k', lw=3, ls='--',
               label=f"1D ({mloss_label}, {vinf_label})"),
    ]
    fig.legend(
        handles=style_handles,
        loc='lower center',
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    # # fig.suptitle(
    # #     f"3D vs 1D abundance comparison — {chemistry}",
    # #     fontsize=15, y=1.01,
    # )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    chemistry = "Orich"  # "Crich" or "Orich"
    pmf = "wind_v10"
    max_particle_id = None
    start_index = 2
    n_select = None
    n_radius = 1600
    show_plot = True
    compare_1d = True  # set False to produce the 3-D-only plot

    use_select_particle_ids = True
    max_particles = None

    source_dir = BASE_PATH / f"evolving_model/{chemistry}/ev_output"
    save_dir = BASE_PATH / f"figures/Evolution_traces/{chemistry}"
    save_dir.mkdir(parents=True, exist_ok=True)

    files = list_particle_files(source_dir)

    if use_select_particle_ids:
        from run_evolving_models import select_particle_ids

        particle_IDs_file = BASE_PATH / "traces" / pmf / "particle_IDs.txt"
        trace_dir = BASE_PATH / "traces" / pmf / "trace_output_with_av"
        particle_IDs = select_particle_ids(
            particle_IDs_file,
            trace_dir,
            max_particle_id=max_particle_id,
            start_index=start_index,
            n_select=n_select,
            min_data_rows=3,
        )

        # Custom overrides – adjust as needed
        # if len(particle_IDs) > 1:
        #     particle_IDs[1] = 29823
        # if len(particle_IDs) > 2:
        #     particle_IDs[2] = 46371
        # particle_IDs = [29823, 46371]  # uncomment to override list entirely

        files = files_from_particle_ids(source_dir, particle_IDs)

    if max_particles is not None:
        files = files[:max_particles]

    if not files:
        raise FileNotFoundError(f"No files matching ev_*.dat found in: {source_dir}")

    all_data = load_particle_data(files)
    r_grid = common_radius_grid(all_data, n_points=n_radius)

    species_to_average = list(dict.fromkeys(parents + daughters_Crich + daughters_Orich))
    averages, contributors = average_species(all_data, species_to_average, r_grid)

    if compare_1d:
        output_file = save_dir / f"ab_mean_compare1D.pdf"
        plot_avg_abundances_compare_1d(
            r_grid,
            averages,
            contributors,
            chemistry,
            output_file,
            show=show_plot,
        )

        output_file = save_dir / f"ab_mean_compare1D_grid.pdf"
        plot_compare_1d_grid(
            r_grid,
            averages,
            contributors,
            chemistry,
            output_file,
            show=show_plot,
        )
    else:
        output_file = save_dir / f"ab_mean.pdf"
        plot_avg_abundances(
            r_grid,
            averages,
            contributors,
            chemistry,
            output_file,
            show=show_plot,
        )

    print(f"Loaded particles: {len(all_data)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()