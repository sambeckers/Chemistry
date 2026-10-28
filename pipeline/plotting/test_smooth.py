# import numpy as np
# import matplotlib.pyplot as plt
# import render_slice_v2 as rs

# XLIM = 6000
# VMIN, VMAX = 1e-9, 1e-1          # abundance limits (real values, not exponents)

# phantom, h5 = rs.resolve_dump_pair(rs.DUMP_DIR, rs.PHANTOM_DIR, None, 1581)
# sdf, meta = rs.prepare_render_dataframe(phantom, h5, "CO", fraction=1.0, col_dens=False)

# def get_img(normalize, npix, xsec=0):
#     """Let Sarracen render into a throwaway axes and return the raw array."""
#     fig, ax = plt.subplots()
#     sdf.render("CO", x="x", y="z",
#                xlim=(-XLIM, XLIM), ylim=(-XLIM, XLIM), normalize=normalize,
#                x_pixels=npix, y_pixels=npix,
#                log_scale=False, cbar=False, ax=ax)
#     img = np.array(ax.images[0].get_array(), dtype=float)
#     plt.close(fig)
#     return np.nan_to_num(img, nan=0.0)      # same as plons remove_nans

# def show(ax, img, title, gouraud=False):
#     n = img.shape[0]
#     g = np.linspace(-XLIM, XLIM, n)
#     X, Y = np.meshgrid(g, g)
#     obs = np.log10(img + 1e-99)             # same as plons plotSlice
#     ax.pcolormesh(X, Y, obs, cmap="gist_heat",
#                   vmin=np.log10(VMIN), vmax=np.log10(VMAX),
#                   shading="gouraud" if gouraud else "auto")
#     ax.set_aspect("equal"); ax.set_facecolor("k"); ax.set_title(title)

# fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
# show(axes[0], get_img(True,  512), "slice, normalize=True, 512 px")
# show(axes[1], get_img(True,  200), "slice, normalize=True, 200 px, gouraud", gouraud=True)
# show(axes[2], get_img(False, 512), "slice, normalize=False, 512 px")
# plt.tight_layout(); plt.show()

import numpy as np, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import render_slice_v2 as rs

XLIM = 6000
phantom, h5 = rs.resolve_dump_pair(rs.DUMP_DIR, rs.PHANTOM_DIR, None, 1581)
sdf, meta = rs.prepare_render_dataframe(phantom, h5, "CO", fraction=1.0)

coords = sdf[["x", "y", "z"]].to_numpy()
vals   = sdf["CO"].to_numpy(dtype=float)
tree   = cKDTree(coords)
_, idx = tree.query(coords, k=64)                  # 32 nearest neighbours incl. self
sdf["CO_smooth"] = 10 ** np.log10(vals[idx]).mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for ax, q in zip(axes, ["CO", "CO_smooth"]):
    rs.render_quantity(sdf, q, plane="xz", ax=ax, xlim=XLIM,
                       xsec=None, vmin=1e-9, vmax=4e-4)
    ax.set_title(q)
plt.show()