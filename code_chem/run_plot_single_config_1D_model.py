from __future__ import division
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import importlib
from kb.modeling.tools import CodeIO
from kb import KoekenBak
from kb.managers import PlottingManager as PlottingManagerModule
from kb.plotting import plotFunctions as plotFunctionsModule
from kb.plotting.objects import PlotChem as PlotChemModule
from kb import path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'plotting'))
from n_distinct_colours import generate_colormap
from config import BASE_PATH, parents, daughters_Crich as daughters, daughters_Orich

# Output folder where the figure will be saved
savedirmain = BASE_PATH
CRICH = True
if CRICH:
    model_folder = 'output_1D/complete_1D_model_Crich'
    inputfile = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot_Vinf_Crich.dat')
    out = 'Crich'
else:
    model_folder = 'output_1D/complete_1D_model_Orich'
    inputfile = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot_Vinf_Orich.dat')
    out = 'Orich'

# Available model labels (same ordering as model.star_grid)
MLOSS = [
    "$\\dot{M}=10^{-5} M_{\\odot} \\rm{yr}^{-1}$",
    "$\\dot{M}=10^{-6} M_{\\odot} \\rm{yr}^{-1}$",
    "$\\dot{M}=10^{-7} M_{\\odot} \\rm{yr}^{-1}$",
]
VELOCITY = [
    "$v_{\\infty}=15 \\rm{km\ s}^{-1}$",
    "$v_{\\infty}=5 \\rm{km\ s}^{-1}$",
    "$v_{\\infty}=5 \\rm{km\ s}^{-1}$",
]

# Choose one configuration here
TARGET_MLOSS = MLOSS[2]
TARGET_VELOCITY = VELOCITY[2]

# Parent and daughter molecules (loaded from config.py)
parents = parents
daughters = daughters
# daughters = daughters_Orich        (O-rich, uncomment import alias to switch)


def run_model(kb_inputfile):
    importlib.reload(plotFunctionsModule)
    importlib.reload(PlotChemModule)
    importlib.reload(PlottingManagerModule)
    importlib.reload(KoekenBak)
    model = KoekenBak.KoekenBak(kb_inputfile)
    model.startSession()
    return model


def find_star_for_config(model, target_mloss, target_velocity):
    for idx, (star, mloss_label, vinf_label) in enumerate(zip(model.star_grid, MLOSS, VELOCITY)):
        if mloss_label == target_mloss and vinf_label == target_velocity:
            return idx, star, mloss_label, vinf_label

    available = [f"{m}, {v}" for m, v in zip(MLOSS, VELOCITY)]
    raise ValueError(
        "Requested configuration not found. "
        f"Requested: {target_mloss}, {target_velocity}. "
        f"Available: {available}"
    )


def get_fractional_abundance(fracs, molecule):
    try:
        return fracs[molecule]
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def get_distinct_colors(n_colors):
    if n_colors <= 0:
        return np.empty((0, 4))
    min_safe = 14
    cmap = generate_colormap(max(n_colors, min_safe))
    return np.array(cmap.colors[:n_colors])


def plot_single_configuration():
    model = run_model(inputfile)
    idx, star, mloss_label, vinf_label = find_star_for_config(model, TARGET_MLOSS, TARGET_VELOCITY)

    folder = os.path.join(path.cout, 'models', star['LAST_CHEMISTRY_MODEL']) + '/'
    print(f"Using model folder: {folder}")

    radius = CodeIO.getChemistryPhysPar(folder + 'csphyspar_smooth.out', 'RADIUS')
    fracs = CodeIO.getChemistryAbundances(folder + 'csfrac_smooth.out')

    parent_groups = np.array_split(parents, 3)

    colors_parents = get_distinct_colors(len(parents))
    colors_daughters = get_distinct_colors(len(daughters))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300, sharex=True, sharey=True)
    parent_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    daughter_ax = axes[1, 1]

    parent_title_suffixes = ['I', 'II', 'III']

    for group_idx, (ax, parent_group) in enumerate(zip(parent_axes, parent_groups)):
        for mol in parent_group:
            color_idx = parents.index(mol)
            frac = get_fractional_abundance(fracs, mol)
            if frac is None:
                print(f"Skipping missing parent molecule: {mol}")
                continue
            ax.loglog(radius, frac, color=colors_parents[color_idx], linewidth=1.2, label=mol)

        ax.set_title(f"Parents {parent_title_suffixes[group_idx]}")
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(loc='best', fontsize='x-small', ncol=2)

    for i, mol in enumerate(daughters):
        frac = get_fractional_abundance(fracs, mol)
        if frac is None:
            print(f"Skipping missing daughter molecule: {mol}")
            continue
        daughter_ax.loglog(
            radius,
            frac,
            color=colors_daughters[i % len(colors_daughters)],
            linewidth=1.2,
            label=mol,
        )

    daughter_ax.set_title('Daughters')
    daughter_ax.grid(True, which='both', alpha=0.25)
    daughter_ax.legend(loc='best', fontsize='x-small', ncol=2)

    for ax in axes[1, :]:
        ax.set_xlabel('Radius [cm]')
    for ax in axes[:, 0]:
        ax.set_ylabel('Fractional abundance')

    fig.suptitle(f"{mloss_label}, {vinf_label}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outname = f"fracab_{out}_{idx}.pdf"
    outpath = savedirmain / 'figures' / '1D_model' / outname
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    print(f"Saved: {outpath}")
    plt.show()


if __name__ == '__main__':
    plot_single_configuration()
