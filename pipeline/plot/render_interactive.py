#!/usr/bin/env python3
"""
render_interactive.py
@author(s): Sam Beckers

Interactive script for SPHRenderer() in render_slice.py
"""
import importlib
import sys
from pathlib import Path
import render_slice
importlib.reload(render_slice)          # pick up edits to render_slice.py
from render_slice import SPHRenderer

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT / "scripts"))
from common import PipelineConfigManager
_PIPELINE_CFG = PipelineConfigManager.load(_PIPELINE_ROOT / "config" / "pipeline_config.yaml")

# Setup ---------------------------------------------------------------
# decades=4 -> every plot gets vmin = vmax / 10**4 automatically
r = SPHRenderer(
    save_dir_base=Path("/fred/oz304/marievds/figures/v10a09_out"),
    dump_dir=_PIPELINE_CFG["paths"]["final_output_dir"],
    phantom_dir=_PIPELINE_CFG["paths"]["phantom_dump_dir"],
    chemistry="Crich",
    dark_mode=False,
    decades=3,
    fast=1,
    # smooth_au=150,
)

# r.list_dumps()
r.set_dump(index=1581)
# r.set_dump(path="/fred/oz304/beckers/v10a09_out/output/dumps/dump_XXXX.h5")

# for mol in r.PARENTS + r.DAUGHTERS:
#     r.plot_single_render(mol, plane="xy", xlim=6000, dens_weight=True, 
#     show=True, save=True, contours=True)
#     # radii_save_path="/fred/oz304/beckers/v10a09_out/output/radii_contours.txt")

# for mol in ['C', 'O', 'N']:
#     r.plot_single_render(mol, plane="xz", xlim=750, dens_weight=True, 
#     show=True,)
#     r.plot_single_render(mol, plane="xy", xlim=750, dens_weight=True, 
#     show=True,)

# Single render
fig = r.plot_single_render(
    "CO",
    plane="xz",
    xlim=3000,
    # dens_weight=True,
    col_dens=True,
    show=True,
    contours=True,
    contour_fracs=(0.5,),
    save=True,                      # use the standard directory/file naming
)

# Other ways to set the colour range ----------------------------------
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, decades=3)       # 3 decades
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, vmax=1e-3)       # vmin = vmax/1e4
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, vmin=1e-9, vmax=1e-4)
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, decades=0)       # autoscale (off)

# Compare particle fractions (shared colour range) --------------------
# fig = r.plot_fraction_compare("CO", plane="xz", xlim=6000, dens_weight=True)               # 3 panels
# fig = r.plot_fraction_compare("CO", plane="xz", xlim=6000, dens_weight=True, nine=True)    # 9 panels

# Physical overview ---------------------------------------------------
# fig = r.plot_phys_overview(xlim=200)

# #  Velocity components + wind cone -------------------------------------
# fig = r.plot_velocity_components(plane="xz", xlim=750, wind_angle=True, v_wind_kms=10.0)

# Parent / daughter ---------------------------------------------------
# for parent, daughter in r.PARENT_DAUGHTER_PAIRS:
#     r.plot_parent_daughter_pair(parent, daughter, plane="xz", xlim=5000, dens_weight=True, decades=4)

# Molecule grid -------------------------------------------------------
# For publication:
# fig = r.plot_molecule_grid(r.PARENTS, plane="xz", xlim=750, dens_weight=True, decades=3, save=True)
# fig = r.plot_molecule_grid(r.DAUGHTERS, plane="xz", xlim=500, dens_weight=True, decades=3, save=True)

# fig = r.plot_molecule_grid(r.DAUGHTERS, plane="xz", xlim=500, dens_weight=True, decades=3, save=True)
# fig = r.plot_molecule_grid(r.DAUGHTERS, plane="xy", xlim=500, dens_weight=True, decades=3, save=True)
# fig = r.plot_molecule_grid(r.DAUGHTERS, plane="xz", xlim=200, dens_weight=True, decades=3, save=True)
# fig = r.plot_molecule_grid(r.DAUGHTERS, plane="xy", xlim=200, dens_weight=True, decades=3, save=True)
# fig = r.plot_molecule_grid(r.PARENTS, plane="xz", xlim=6000, dens_weight=True, contours=True, save=True)
# fig = r.plot_molecule_grid(r.DAUGHTERS, plane="xz", xlim=6000, dens_weight=True, contours=True, save=True)

# Contours + radii table ----------------------------------------------
# fig = r.plot_single_render(
#     "CO", plane="xy", xlim=6000, dens_weight=True, contours=True,
#     radii_save_path="/fred/oz304/beckers/v10a09_out/output/radii_contours.txt",
# )

# Column-density weighting (abundance x n_H2) --------------------------
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, col_dens=True)

# Loop over several molecules -----------------------------------------
# for mol in ["CO", "HCN", "SiO"]:
#     r.plot_single_render(mol, plane="xz", xlim=6000, dens_weight=True, save=True, show=False)