# %%
"""
render_interactive.py

Interactive driver for sph_render.SPHRenderer.
Run cell-by-cell (VS Code / Spyder / Jupyter `# %%` cells).
"""
import importlib
import render_slice_v3
importlib.reload(render_slice_v3)          # pick up edits to sph_render.py
from render_slice_v3 import SPHRenderer

# Setup ---------------------------------------------------------------
# decades=4 -> every plot gets vmin = vmax / 10**4 automatically
r = SPHRenderer(
    dump_dir="/fred/oz304/beckers/v10a09_out/output/dumps",
    phantom_dir="/fred/oz304/beckers/v10a09",
    chemistry="Crich",
    dark_mode=False,
    decades=3,
    fast=0.1,
)

# r.list_dumps()
r.set_dump(index=1581)
# r.set_dump(path="/fred/oz304/beckers/v10a09_out/output/dumps/dump_XXXX.h5")

# Single render
# fig = r.plot_single_render(
#     "CH2",
#     plane="xz",
#     xlim=5000,
#     dens_weight=True,
#     show=True,
#     # save=True,                      # use the standard directory/file naming
# )

# Other ways to set the colour range ----------------------------------
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, decades=3)       # 3 decades
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, vmax=1e-3)       # vmin = vmax/1e4
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, vmin=1e-9, vmax=1e-4)
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, dens_weight=True, decades=0)       # autoscale (off)

# Compare particle fractions (shared colour range) --------------------
# fig = r.plot_fraction_compare("CO", plane="xz", xlim=6000, dens_weight=True)               # 3 panels
# fig = r.plot_fraction_compare("CO", plane="xz", xlim=6000, dens_weight=True, nine=True)    # 9 panels

# Physical overview ---------------------------------------------------
# fig = r.plot_phys_overview(xlim=6000)

#  Velocity components + wind cone -------------------------------------
# fig = r.plot_velocity_components(plane="xz", xlim=6000, wind_angle=True, v_wind_kms=10.0)

# Parent / daughter ---------------------------------------------------
# for parent, daughter in r.PARENT_DAUGHTER_PAIRS:
#     r.plot_parent_daughter_pair(parent, daughter, plane="xz", xlim=5000, dens_weight=True, decades=4)

# Molecule grid -------------------------------------------------------
fig = r.plot_molecule_grid(r.PARENTS, plane="xz", xlim=750, dens_weight=True)
# fig = r.plot_molecule_grid(r.DAUGHTERS, plane="xz", xlim=500, dens_weight=True)

# %% Contours + radii table ----------------------------------------------
# fig = r.plot_single_render(
#     "CO", plane="xy", xlim=6000, dens_weight=True, contours=True,
#     radii_save_path="/fred/oz304/beckers/v10a09_out/output/radii_contours.txt",
# )

# %% Column-density weighting (abundance x n_H2) --------------------------
# fig = r.plot_single_render("CO", plane="xz", xlim=6000, col_dens=True)

# %% Loop over several molecules -----------------------------------------
# for mol in ["CO", "HCN", "SiO"]:
#     r.plot_single_render(mol, plane="xz", xlim=6000, dens_weight=True, save=True, show=False)