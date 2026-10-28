#!/usr/bin/env python3
"""
smoothmesh_slice_v3.py

Loads the model exactly like SlicePlot_nCO.ipynb (plons.LoadSetup +
plons.LoadFullDump -> dumpData in cgs), adds the CO fields from your HDF5
chemistry dump (matched on particle id), and plots with the notebook's
display (NaN -> 0, log10(x + 1e-99), pcolormesh gouraud, white contours).

Two ways to get the 2D field from the same dumpData:

  --method plons     plons.SmoothingKernelScript.smoothMesh on the notebook's
                     mesh (Y = 0).  plons only offers this plane evaluation;
                     it has no line-of-sight projection.
  --method sarracen  line-of-sight projection (xsec=None, normalize=True),
                     the same as render_slice_v2.py.
                     --dens-weight  density-weighted average
  --col-dens         plot n_X = abundance * n [m^-3] instead of abundance

Examples
    python smoothmesh_slice_v3.py --dump-index 1581 --method plons --col-dens
    python smoothmesh_slice_v3.py --dump-index 1581 --method sarracen --col-dens
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import plons
import plons.SmoothingKernelScript as sk
import plons.ConversionFactors_cgs as cgs

DUMP_DIR = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
PHANTOM_DIR = Path("/fred/oz304/beckers/v10a09")
SETUP_DIR = PHANTOM_DIR        # folder with wind.in / wind.setup (notebook: <model>/Model/)
PREFIX = "wind"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def normalise_name(name: str) -> str:
    c = name.strip().lower()
    c = c.replace("+", "_plus").replace("-", "_minus").replace("/", "_")
    c = c.replace("(", "_").replace(")", "_").replace(".", "_")
    c = re.sub(r"[^a-z0-9_]+", "_", c)
    return re.sub(r"_+", "_", c).strip("_")


def resolve_dump_pair(dump_index: int):
    h5 = sorted(DUMP_DIR.glob("dump_*.h5"))[dump_index]
    stem = h5.stem.split("_", 1)[1]
    return PHANTOM_DIR / f"{PREFIX}_{stem}", h5


def load_dump_data(phantom_path: Path, h5_path: Path, species: str):
    """plons dumpData (cgs) + 'abund' and 'nX' [m^-3] from the HDF5 dump."""
    setup = plons.LoadSetup(str(SETUP_DIR), PREFIX)
    dumpData = plons.LoadFullDump(str(phantom_path), setup)

    ids = dumpData["iorig"].to_numpy(dtype=np.int64)
    with h5py.File(h5_path, "r") as f:
        h5_ids = f["particles/id"][:].astype(np.int64)
        ab_raw = f["particles"][normalise_name(species)][:]
        dens_raw = f["particles/density"][:]            # cm^-3

    order = np.argsort(h5_ids)
    sorted_ids = h5_ids[order]
    ins = np.clip(np.searchsorted(sorted_ids, ids), 0, len(sorted_ids) - 1)
    matched = sorted_ids[ins] == ids

    ab = np.where(matched, ab_raw[order][ins], np.nan)
    dens = np.where(matched, dens_raw[order][ins], np.nan)
    good = np.isfinite(ab) & (ab > 0) & (ab <= 1) & np.isfinite(dens) & (dens > 0)
    print(f"Particles: {len(ids):,}; usable: {good.sum():,}")

    dumpData["abund"] = np.where(good, ab, 0.0)
    dumpData["nX"] = np.where(good, ab * dens * 1e6, 0.0)    # cm^-3 -> m^-3
    return dumpData[good]


# ---------------------------------------------------------------------------
# Field computation
# ---------------------------------------------------------------------------
def plons_field(dumpData, keys, lims_au, n, neighbours=100):
    """Exactly the notebook's do(): smoothMesh on X, Y=0, Z (cm)."""
    g = np.linspace(-lims_au, lims_au, n) * cgs.au
    X, Z = np.meshgrid(g, g)
    Y = np.zeros_like(X)
    out = {}
    for k in keys:
        s = sk.smoothMesh(X, Y, Z, dumpData, [k], neighbours)[k]
        out[k] = np.nan_to_num(s, nan=0.0)                   # plons remove_nans
    return out


def sarracen_field(dumpData, keys, lims_au, n, dens_weight):
    """Sarracen render, xsec=None, normalize=True. Array indexed [z, x].

    dumpData is in cgs, so limits are given in cm. With normalize=True the
    unit scaling of mass / rho cancels, so the result is unit independent.
    """
    lim = lims_au * cgs.au
    out = {}
    for k in keys:
        fig, ax = plt.subplots()
        dumpData.render(k, x="x", y="z",
                        xlim=(-lim, lim), ylim=(-lim, lim),
                        x_pixels=n, y_pixels=n,
                        xsec=None, normalize=True, dens_weight=dens_weight,
                        log_scale=False, cbar=False, ax=ax)
        out[k] = np.nan_to_num(np.array(ax.images[0].get_array(), dtype=float), nan=0.0)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot (plotSlice from the notebook)
# ---------------------------------------------------------------------------
def plot_field(ax, X, Z, field, clim, cmap="gist_heat"):
    ax.set_aspect("equal")
    ax.set_facecolor("k")
    return ax.pcolormesh(X, Z, np.log10(field + 1e-99), cmap=cmap,
                         vmin=clim[0], vmax=clim[1],
                         shading="gouraud", rasterized=True)


def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


INTERACTIVE_ARGS = [          # edit when running in Jupyter / IPython / Spyder
    "--dump-index", "1581",
    "--quantity", "CO",
    "--method", "plons",      # or "sarracen"
    # "--col-dens",
    # "--dens-weight",
    "--lims", "6000",         # mesh half-width, as in the notebook
    "--axlim", "5500",        # plotted half-width, as in the 3x3 figure
    "--fac", "200",           # notebook: 200 for v05*/v10*, 100 for v20*
    "--n", "600",
    "--show",
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dump-index", type=int, default=1581)
    p.add_argument("--quantity", default="CO")
    p.add_argument("--method", choices=["plons", "sarracen"], default="plons")
    p.add_argument("--col-dens", action="store_true",
                   help="plot n_X = abundance * n [m^-3]")
    p.add_argument("--dens-weight", action="store_true", help="sarracen method only")
    p.add_argument("--lims", type=float, default=6000, help="half-width [au]")
    p.add_argument("--axlim", type=float, default=None,
                   help="plotted half-width [au]; default = --lims (notebook: mesh 6000, axes 5500)")
    p.add_argument("--n", type=int, default=600, help="pixels per axis")
    p.add_argument("--neighbours", type=int, default=100, help="plons method only")
    p.add_argument("--fac", type=float, default=100,
                   help="notebook clim for n_X: vmin = log10(max)/fac")
    p.add_argument("--decades", type=float, default=None,
                   help="override: vmin = log10(max) - decades")
    p.add_argument("--out", default="smoothmesh_slice_v3.png")
    p.add_argument("--show", action="store_true")
    if argv is None and is_interactive():
        argv = INTERACTIVE_ARGS
    a = p.parse_args(argv)

    phantom, h5 = resolve_dump_pair(a.dump_index)
    print(f"Phantom: {phantom.name}   HDF5: {h5.name}")
    dumpData = load_dump_data(phantom, h5, a.quantity)
    key = "nX" if a.col_dens else "abund"

    g = np.linspace(-a.lims, a.lims, a.n)
    X, Z = np.meshgrid(g, g)                                  # au

    if a.method == "plons":
        out = plons_field(dumpData, ["abund", "nX"], a.lims, a.n, a.neighbours)
    else:
        out = sarracen_field(dumpData, ["abund", "nX"], a.lims, a.n, a.dens_weight)
    field, ratio = out[key], out["abund"]

    lmax = np.log10(np.amax(field))
    if a.decades is not None:
        clim = (lmax - a.decades, lmax)
    elif key == "nX":
        clim = (lmax / a.fac, lmax)               # the notebook's choice
    else:
        clim = (lmax - 6, lmax)
    print(f"log10 colour limits: {clim}")

    fig, ax = plt.subplots(figsize=(6, 5))
    m = plot_field(ax, X, Z, field, clim)
    # ax.contour(X, Z, ratio, levels=[ratio.max() / 100, ratio.max() / 2],
    #            colors="white", linestyles=[":", "-"], linewidths=[1.3, 0.7])
    L = a.axlim if a.axlim is not None else a.lims
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_xlabel(r"$x$ [au]")
    ax.set_ylabel(r"$z$ [au]")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"$\log_{10}(n^{\rm CO})$ [m$^{-3}$]" if key == "nX"
                 else r"$\log_{10}$ abundance rel. to H$_2$")
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(f"Saved {a.out}")
    if a.show:
        plt.show()
    return dumpData, out, (X, Z)


if __name__ == "__main__":
    main()