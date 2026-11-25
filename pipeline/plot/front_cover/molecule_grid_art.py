#!/usr/bin/env python3
"""
molecule_grid_art.py

Render all available molecular column densities from a chemistry dump
into a dense, annotation‑only grid suitable for A4 artwork.

Two modes:
  panels   – one array task renders *one* molecule panel.
  assemble – stitch all pre‑rendered panels into an A4 grid.

Designed for SLURM array jobs (≈30 s per molecule typical).
Requires Pillow for assembly (imported as PIL.Image).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sarracen
from astropy import units as u          # <<< added
from matplotlib.colors import Normalize  # (LogNorm removed, not used)

# ---- local helpers ----------------------------------------------------------

BASE_PATH = Path("/fred/oz304/beckers/Chemistry")

# We import plot_utils if available, but can live without it
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "plot_utils", BASE_PATH / "code_chem" / "plotting" / "plot_utils.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    set_plot_style = _mod.set_plot_style
    format_species_label = _mod.format_species_label
except Exception:
    def set_plot_style(dark: bool) -> None:
        if dark:
            plt.style.use("dark_background")
    def format_species_label(name: str) -> str:
        return f"${name}$"


# ---- Paths ------------------------------------------------------------------

DUMP_DIR = Path("/fred/oz304/beckers/v10a09_out/output/dumps")
PHANTOM_DIR = Path("/fred/oz304/beckers/v10a09")
SAVE_DIR_BASE = BASE_PATH / "figures/v10a09_out/molecule_art"

# ---- Known HDF5 keys that are NOT molecular abundances -----------------------
NON_SPECIES_KEYS = {
    "id", "density", "temp", "av", "x", "y", "z", "vx", "vy", "vz",
    "h", "rho", "u", "alpha", "divv", "dt", "iorig", "itype", "m",
    "p", "particle_id", "porig",
}

# ---- Column‑density weighting constant --------------------------------------
CM3_TO_M3 = (1 * u.cm**-3).to(u.m**-3).value   # now uses astropy.units

# ---- Panel dimension --------------------------------------------------------
PANEL_SIZE_INCH = 0.45          # width = height of a single panel (square)
A4_LANDSCAPE = (11.69, 8.27)   # inches (297 x 210 mm)
A4_PORTRAIT  = (8.27, 11.69)

# ---- Global font size for annotations ---------------------------------------
ANNOT_FONT_SIZE = 4

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
def truncate_colormap(cmap_name="gist_heat", minval=0.0, maxval=0.95, n=256):
    cmap = cm.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
        f"trunc_{cmap_name}",
        cmap(np.linspace(minval, maxval, n))
    )
GIST_HEAT_NO_WHITE = truncate_colormap("gist_heat", maxval=0.95)
# ============================================================================
# Molecule discovery
# ============================================================================

def get_all_species(dump_path: Path) -> List[str]:
    """Return sorted list of all molecular species in the HDF5 file."""
    with h5py.File(dump_path, "r") as f:
        keys = set(f["particles"].keys())
    species = sorted(keys - NON_SPECIES_KEYS)
    return species

# ---- Species name recovery --------------------------------------------------

CHEMISTRY_ROOT = Path("/fred/oz304/beckers/Chemistry/evolving_model")

UNDEFINED_AB_SPECIES = {
    "F+",
    "COOCH3+",
    "C2H4CN",
    "HC2O",
    "HCCN",
    "CH3COOH+",
    "COOCH3",
    "CH3COOH2+",
    "CH3COOH",
    "CH3CO",
    "COOH",
    "He+",
    "HF+",
}


def normalise_species_name(name: str) -> str:
    """
    Reproduce the original pipeline normalisation exactly.
    """
    cleaned = name.strip().lower()
    cleaned = cleaned.replace("+", "_plus")
    cleaned = cleaned.replace("-", "_minus")
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace("(", "_")
    cleaned = cleaned.replace(")", "_")
    cleaned = cleaned.replace(".", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def build_species_lookup() -> dict[str, str]:
    """
    Build mapping:
        c2_minus -> C2-
        c3_plus  -> C3+
        c2h4     -> C2H4
        ...
    from all available .specs files.
    """
    lookup = {}
    try:
        with open(CHEMISTRY_ROOT / "rate12_complex_atomic_Crich.specs", "r", encoding="ascii") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 2:
                    continue

                species = parts[1]

                # Apply the same filtering used in the pipeline
                if species.startswith("G"):
                    continue
                if "Y" in species:
                    continue
                if species in UNDEFINED_AB_SPECIES:
                    continue

                lookup[normalise_species_name(species)] = species

    except Exception as e:
            print(f"Warning: failed to read .specs file: {e}", file=sys.stderr)

    print(f"Loaded {len(lookup)} species names from chemistry catalog")
    return lookup


SPECIES_LOOKUP = build_species_lookup()

def recover_species_name(key: str) -> str:
    """
    Recover original species name from a normalised HDF5 key.

    Falls back to a simple denormalisation if the key is not found.
    """
    if key in SPECIES_LOOKUP:
        return SPECIES_LOOKUP[key]

    # fallback
    return (
        key.replace("_plus", "+")
           .replace("_minus", "-")
           .replace("_", "")
    )
# ============================================================================
# Render a single panel (no axes, no cbar, only annotation)
# ============================================================================

def render_single_panel(
    phantom_path: Path,
    dump_path: Path,
    species: str,
    plane: str = "xy",
    xlim: float = 100.0,
    cmap: str = "gist_heat",
    dark: bool = True,
) -> plt.Figure:
    """
    Create a square panel figure (no margins) showing the column density
    of *species* with a white top‑right annotation of its name.
    """
    # ---- Load SPH data and attach column density ----------------------------
    sdf, _ = sarracen.read_phantom(phantom_path)
    with h5py.File(dump_path, "r") as f:
        h5_ids = f["particles/id"][:].astype(np.int64)
        dens = f["particles/density"][:]   # n_H2 [cm^-3]
        abun = f["particles"][species][:]  # abundance (dimensionless)

    # Match particles
    sdf_ids = sdf["iorig"].to_numpy(dtype=np.int64)
    sort_idx = np.argsort(h5_ids)
    h5_sorted = h5_ids[sort_idx]
    ins = np.searchsorted(h5_sorted, sdf_ids)
    ins_clipped = np.clip(ins, 0, len(h5_sorted) - 1)
    matched = h5_sorted[ins_clipped] == sdf_ids

    dens_matched = np.where(matched, dens[sort_idx][ins_clipped], np.nan)
    abun_matched = np.where(matched, abun[sort_idx][ins_clipped], np.nan)
    # Keep only valid particles (finite, >0, <=1)
    valid = np.isfinite(abun_matched) & (abun_matched > 0) & (abun_matched <= 1)
    col_dens = np.where(valid, abun_matched * dens_matched * CM3_TO_M3, np.nan)
    sdf["col_dens"] = col_dens

    sdf = sdf.dropna(subset=["col_dens"])
    if len(sdf) == 0:
        raise RuntimeError(f"No valid particles for {species}")

    # ---- Render -------------------------------------------------------------
    set_plot_style(dark)

    fig, ax = plt.subplots(figsize=(PANEL_SIZE_INCH, PANEL_SIZE_INCH), dpi=300)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Sarracen render (column‑density, log scale, no colour bar)
    sdf.render(
        "col_dens",
        x="x", y="y" if plane == "xy" else "z",
        log_scale=True,
        cmap=cmap,
        cbar=False,
        ax=ax,
        xlim=(-xlim, xlim), ylim=(-xlim, xlim),
        normalize=True,
    )

    ax.set_facecolor("black")

    # Tiny inner ticks, no labels
    ax.tick_params(
        direction="in",
        length=1, width=0.5,
        colors="white",
        labelleft=False, labelbottom=False,
        top=True, right=True,
        which="both",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # # Remove axis spines entirely
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # Annotation: species name top right, white, small font
    label = format_species_label(recover_species_name(species))
    ax.text(
        0.95, 0.95, label,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=ANNOT_FONT_SIZE,
        color="white",
        fontweight="bold",
    )

    return fig


# ============================================================================
# Assembly: combine panel PNGs into A4 grid
# ============================================================================

def assemble_grid(
    panel_dir: Path,
    output_stem: str,
    orientation: str = "both",
) -> None:
    """
    Read all panel PNGs from *panel_dir*, compute a grid layout that fits
    A4 paper, and save a combined figure as PDF and PNG.
    """
    from PIL import Image        # Pillow, imported as PIL.Image

    panel_files = sorted(panel_dir.glob("panel_*.png"))
    if not panel_files:
        print(f"No panels found in {panel_dir}")
        return

    if orientation in ("landscape", "both"):
        fig_w, fig_h = A4_LANDSCAPE
        ncols = int(fig_w / PANEL_SIZE_INCH)
        nrows = int(fig_h / PANEL_SIZE_INCH)
        total = ncols * nrows
        panels_to_use = panel_files[:total]
        print(f"Landscape grid: {ncols} cols x {nrows} rows = {total} panels")
        _assemble_one_grid(panels_to_use, ncols, nrows,
                           figsize=(fig_w, fig_h),
                           output=f"{output_stem}_landscape")

    if orientation in ("portrait", "both"):
        fig_w, fig_h = A4_PORTRAIT
        ncols = int(fig_w / PANEL_SIZE_INCH)
        nrows = int(fig_h / PANEL_SIZE_INCH)
        total = ncols * nrows
        panels_to_use = panel_files[:total]
        print(f"Portrait grid: {ncols} cols x {nrows} rows = {total} panels")
        _assemble_one_grid(panels_to_use, ncols, nrows,
                           figsize=(fig_w, fig_h),
                           output=f"{output_stem}_portrait")


def _assemble_one_grid(
    panel_files: List[Path],
    ncols: int,
    nrows: int,
    figsize: tuple,
    output: str,
) -> None:
    """Place images into a matplotlib figure grid, no spacing."""
    from PIL import Image

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize,
                             dpi=300,
                             gridspec_kw={"wspace": 0, "hspace": 0},
                             subplot_kw={"xticks": [], "yticks": []})
    axes = np.atleast_2d(axes)

    for idx, (ax, fpath) in enumerate(zip(axes.flat, panel_files)):
        img = Image.open(fpath)
        ax.imshow(img)
        ax.axis("off")
    for ax in axes.flat[len(panel_files):]:
        ax.axis("off")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    for ext in ["pdf", "png"]:
        path = f"{output}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Molecule grid art")
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- panels mode ---------------------------------------------------------
    p_panels = sub.add_parser("panels", help="Render a single molecule panel")
    p_panels.add_argument("--dump", type=str)
    p_panels.add_argument("--dump-index", type=int)
    p_panels.add_argument("--dump-dir", type=str, default=str(DUMP_DIR))
    p_panels.add_argument("--phantom-dir", type=str, default=str(PHANTOM_DIR))
    p_panels.add_argument("--output-dir", type=str, default=str(SAVE_DIR_BASE / "panels"))
    p_panels.add_argument("--chemistry", choices=["Crich", "Orich"], default="Crich")
    p_panels.add_argument("--plane", choices=["xy", "xz"], default="xy")
    p_panels.add_argument("--xlim", type=float, default=100.0)
    p_panels.add_argument("--cmap", default="gist_heat")
    p_panels.add_argument("--task-id", type=int, required=True,
                          help="0‑based index of this array task")
    p_panels.add_argument("--n-tasks", type=int, required=True,
                          help="Total number of array tasks")

    # --- assemble mode -------------------------------------------------------
    p_assemble = sub.add_parser("assemble", help="Assemble panels into grid")
    p_assemble.add_argument("--panel-dir", type=str, required=True)
    p_assemble.add_argument("--output-stem", type=str, required=True,
                            help="Base name (without extension) for the final file")
    p_assemble.add_argument("--orientation", choices=["landscape","portrait","both"],
                            default="both")
    p_assemble.add_argument("--dump-dir", type=str, default=str(DUMP_DIR))
    p_assemble.add_argument("--phantom-dir", type=str, default=str(PHANTOM_DIR))

    args = parser.parse_args()

    # Resolve dump path
    dump_dir = Path(args.dump_dir)
    phantom_dir = Path(args.phantom_dir)

    if args.mode == "panels":
        if args.dump:
            dump_path = Path(args.dump)
            stem = dump_path.stem.split("_", 1)[1]
            phantom_path = phantom_dir / f"wind_{stem}"
        else:
            dumps = sorted(dump_dir.glob("dump_*.h5"))
            if args.dump_index is None:
                raise ValueError("Provide --dump or --dump-index")
            dump_path = dumps[args.dump_index]
            stem = dump_path.stem.split("_", 1)[1]
            phantom_path = phantom_dir / f"wind_{stem}"

        species = get_all_species(dump_path)
        n_species = len(species)
        print(f"Found {n_species} species in {dump_path.name}")
        task_species = np.array_split(species, args.n_tasks)[args.task_id]

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for s in task_species:
            out_png = out_dir / f"panel_{s}.png"
            if out_png.exists():
                print(f"Skip existing {out_png}")
                continue
            print(f"Rendering {s} ...")
            try:
                fig = render_single_panel(
                    phantom_path, dump_path, s,
                    plane=args.plane,
                    xlim=args.xlim,
                    cmap=args.cmap,
                    dark=True,
                )
                fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                print(f"  Saved {out_png}")
            except Exception as e:
                print(f"  ERROR {s}: {e}")

    elif args.mode == "assemble":
        assemble_grid(
            Path(args.panel_dir),
            args.output_stem,
            args.orientation,
        )


if __name__ == "__main__":
    main()