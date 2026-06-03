#!/usr/bin/env python3
"""
compare_radii_1d_3d.py

Compare photodissociation radii from:
  - 1D KoekenBak model  (R at 0.01× and 0.5× initial abundance)
  - 3D Phantom/Sarracen render  (contour radii saved by render_slice_v2.py)

For each molecule present in both datasets, plots R_1D vs R_3D for each
plane (xy, xz) and each threshold (0.01×, 0.5×).

Usage
-----
python compare_radii_1d_3d.py \
    --radii-file /path/to/contour_radii.txt \
    --chemistry Crich \
    --save-dir  /path/to/figures \
    --show
"""

from __future__ import annotations

import argparse
from astropy import units as u
import contextlib
import io
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Project-level imports — same pattern as render_slice_v2.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_PATH

from code_chem.run_plot_single_config_1D_model import (
    run_model,
    find_star_for_config,
    get_fractional_abundance,
    TARGET_MLOSS,
    TARGET_VELOCITY,
)
from kb.modeling.tools import CodeIO
from kb import path as kb_path

# Shared plot style from the pipeline
from code_chem.plotting.plot_utils import set_plot_style, format_species_label

FONT_SIZE = 14

# Fraction thresholds — must match the column names written by render_slice_v2
THRESHOLDS = {
    "001": 0.01,
    "050": 0.50,
}

# Colours / markers for the two planes
PLANE_STYLE = {
    "xy": dict(color="#4C9BE8", marker="o", label="3D  xy"),
    "xz": dict(color="#E8844C", marker="s", label="3D  xz"),
}

# ===========================================================================
# 1D loader  (verbatim interface from the user's existing code)
# ===========================================================================

def load_1d_data(chemistry: str):
    if chemistry == "Crich":
        inputfile = str(BASE_PATH /
                        "KoekenBak/input/20251015_Sam_Mdot_Vinf_Crich.dat")
    else:
        inputfile = str(BASE_PATH /
                        "KoekenBak/input/20251015_Sam_Mdot_Vinf_Orich.dat")

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        model = run_model(inputfile)
        idx, star, mloss_label, vinf_label = find_star_for_config(
            model, TARGET_MLOSS, TARGET_VELOCITY
        )
        folder = (os.path.join(kb_path.cout, "models",
                               star["LAST_CHEMISTRY_MODEL"]) + "/")
        radius_1d = CodeIO.getChemistryPhysPar(
            folder + "csphyspar_smooth.out", "RADIUS"
        )
        fracs_1d  = CodeIO.getChemistryAbundances(folder + "csfrac_smooth.out")
        coldens   = CodeIO.getColumnDensities(folder + "cscoldens_smooth.out")
        nH2       = CodeIO.getChemistryPhysPar(
            folder + "csphyspar_smooth.out", "n(H2)"
        )

    print(f"Loaded 1D model  ({mloss_label}, {vinf_label})")
    return radius_1d, fracs_1d, coldens, nH2, mloss_label, vinf_label


# ===========================================================================
# Compute 1D photodissociation radii
# ===========================================================================

def compute_1d_radii(
    radius_1d: np.ndarray,
    fracs_1d:  dict,
    molecules:  list[str],
) -> dict[str, dict[str, float]]:
    """
    For each molecule return the radius at which the abundance first drops
    to 0.01× and 0.5× of its initial (innermost shell) value.

    Returns
    -------
    {molecule: {"001": R_AU, "050": R_AU}}
    """
    results: dict[str, dict[str, float]] = {}

    for mol in molecules:
        # fracs_1d keys may differ in capitalisation / normalisation
        # key = _match_key(mol, fracs_1d)
        # if key is None:
        #     print(f"  [1D] '{mol}' not found in 1D model — skipping")
        #     continue
        cm_to_AU = (1 * u.cm).to(u.AU).value
        frac = fracs_1d[mol]
        if len(frac) == 0 or not np.any(np.isfinite(frac) & (frac > 0)):
            continue

        # Initial abundance = value at the innermost shell
        f0 = frac[0]
        if not np.isfinite(f0) or f0 <= 0:
            continue

        radii_mol: dict[str, float] = {}
        for tag, threshold in THRESHOLDS.items():
            target = threshold * f0
            # First index where abundance falls AT or BELOW target
            below = np.where(frac <= target)[0]
            if len(below) == 0:
                radii_mol[tag] = np.nan   # never reaches threshold
            else:
                # Linear interpolation for a smoother estimate
                i = below[0]
                if i == 0:
                    radii_mol[tag] = float(radius_1d[0] * cm_to_AU)
                else:
                    # interpolate between i-1 and i
                    f_hi, f_lo = frac[i - 1], frac[i]
                    r_hi, r_lo = radius_1d[i - 1], radius_1d[i]
                    if f_hi == f_lo:
                        radii_mol[tag] = float(r_lo * cm_to_AU)
                    else:
                        t = (target - f_hi) / (f_lo - f_hi)
                        radii_mol[tag] = float((r_hi + t * (r_lo - r_hi)) * cm_to_AU)
        
        results[mol] = radii_mol

    return results


def _match_key(name: str, d: dict) -> str | None:
    """Case-insensitive lookup of *name* in dict *d*."""
    nl = name.lower()
    for k in d:
        if k.lower() == nl:
            return k
    return None


# ===========================================================================
# Load 3D contour radii from the .txt file written by render_slice_v2.py
# ===========================================================================

def load_3d_radii(path: Path) -> dict[str, dict[str, float]]:
    """
    Read the tab-separated file produced by render_slice_v2.py.

    Expected header:
        molecule  R_xy_001  R_xy_050  R_xz_001  R_xz_050

    Returns
    -------
    {molecule: {"R_xy_001": float, "R_xy_050": float,
                "R_xz_001": float, "R_xz_050": float}}
    """
    import csv

    results: dict[str, dict[str, float]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mol = row["molecule"].strip()
            entry: dict[str, float] = {}
            for col in ("R_xy_001", "R_xy_050", "R_xz_001", "R_xz_050"):
                val = row.get(col, "—").strip()
                entry[col] = float(val) if val not in ("—", "", "nan") else np.nan
            results[mol] = entry

    print(f"Loaded 3D radii for {len(results)} molecules from {path.name}")
    return results


# ===========================================================================
# Plotting
# ===========================================================================

def plot_comparison(
    radii_1d:  dict[str, dict[str, float]],
    radii_3d:  dict[str, dict[str, float]],
    save_dir:  Path | None,
    show:      bool,
    chemistry: str,
):
    """
    One figure per threshold (0.01×, 0.5×), showing R_1D vs R_3D for every
    molecule that appears in both datasets, with xy and xz as separate series.
    Also draws a 1:1 reference line.
    """
    # Molecules present in both datasets
    common = sorted(
        set(radii_1d.keys()) & set(radii_3d.keys()),
        key=lambda s: s.lower(),
    )

    if not common:
        print("No molecules in common between 1D and 3D datasets — nothing to plot.")
        return

    for tag, frac_label in [("001", "0.01×"), ("050", "0.50×")]:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        r1d_vals_xy, r3d_vals_xy = [], []
        r1d_vals_xz, r3d_vals_xz = [], []
        labels = []

        for mol in common:
            r1d = radii_1d[mol].get(tag, np.nan)
            r_xy = radii_3d[mol].get(f"R_xy_{tag}", np.nan)
            r_xz = radii_3d[mol].get(f"R_xz_{tag}", np.nan)

            if np.isnan(r1d):
                continue

            labels.append(mol)
            r1d_vals_xy.append(r1d)
            r3d_vals_xy.append(r_xy)
            r1d_vals_xz.append(r1d)
            r3d_vals_xz.append(r_xz)

        r1d_xy = np.array(r1d_vals_xy)
        r3d_xy = np.array(r3d_vals_xy)
        r1d_xz = np.array(r1d_vals_xz)
        r3d_xz = np.array(r3d_vals_xz)

        # --- 1:1 line ---
        all_finite = np.concatenate([
            r1d_xy[np.isfinite(r1d_xy)],
            r3d_xy[np.isfinite(r3d_xy)],
            r1d_xz[np.isfinite(r1d_xz)],
            r3d_xz[np.isfinite(r3d_xz)],
        ])
        if len(all_finite) == 0:
            plt.close(fig)
            continue

        lim_lo = all_finite.min() * 0.7
        lim_hi = all_finite.max() * 1.4
        ref = np.array([lim_lo, lim_hi])
        ax.plot(ref, ref, color="0.6", lw=1.2, ls="--", zorder=0,
                label="1 : 1")

        # --- xy series ---
        mask_xy = np.isfinite(r3d_xy)
        if mask_xy.any():
            sty = PLANE_STYLE["xy"]
            ax.scatter(
                r1d_xy[mask_xy], r3d_xy[mask_xy],
                color=sty["color"], marker=sty["marker"],
                s=60, zorder=3, label=sty["label"],
            )
            for r1, r3, mol in zip(
                r1d_xy[mask_xy], r3d_xy[mask_xy],
                np.array(labels)[mask_xy],
            ):
                ax.annotate(
                    format_species_label(mol),
                    xy=(r1, r3),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=sty["color"],
                )

        # --- xz series ---
        mask_xz = np.isfinite(r3d_xz)
        if mask_xz.any():
            sty = PLANE_STYLE["xz"]
            ax.scatter(
                r1d_xz[mask_xz], r3d_xz[mask_xz],
                color=sty["color"], marker=sty["marker"],
                s=60, zorder=3, label=sty["label"],
            )
            for r1, r3, mol in zip(
                r1d_xz[mask_xz], r3d_xz[mask_xz],
                np.array(labels)[mask_xz],
            ):
                ax.annotate(
                    format_species_label(mol),
                    xy=(r1, r3),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=8,
                    color=sty["color"],
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal")

        ax.set_xlabel(r"$R_{\rm 1D}$ [AU]", fontsize=FONT_SIZE)
        ax.set_ylabel(r"$R_{\rm 3D}$ [AU]", fontsize=FONT_SIZE)
        # ax.set_title(
        #     rf"Photodissociation radius at {frac_label} initial abundance"
        #     f"\n({chemistry})",
        #     fontsize=FONT_SIZE,
        # )

        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, right=True, labelsize=FONT_SIZE * 0.8)
        ax.legend(fontsize=FONT_SIZE * 0.8, framealpha=0.7)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)

        plt.tight_layout()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = save_dir / f"radii_compare_{chemistry}_{tag}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_radii_per_molecule(
    radii_1d:  dict[str, dict[str, float]],
    radii_3d:  dict[str, dict[str, float]],
    save_dir:  Path | None,
    show:      bool,
    chemistry: str,
):
    """
    Alternative view: one panel per molecule, showing both thresholds and
    both planes as grouped bar / scatter so you can read off actual AU values.
    """
    common = sorted(
        set(radii_1d.keys()) & set(radii_3d.keys()),
        key=lambda s: s.lower(),
    )
    if not common:
        return

    n = len(common)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        dpi=150,
        squeeze=False,
    )

    for spare in axes.flat[n:]:
        spare.set_visible(False)

    x = np.arange(2)                          # 0 = 0.01×, 1 = 0.5×
    width = 0.25
    tags = ["001", "050"]
    xlabels = ["0.01×", "0.50×"]

    for idx, mol in enumerate(common):
        ax = axes[idx // ncols][idx % ncols]

        r1d  = [radii_1d[mol].get(t, np.nan) for t in tags]
        r_xy = [radii_3d[mol].get(f"R_xy_{t}", np.nan) for t in tags]
        r_xz = [radii_3d[mol].get(f"R_xz_{t}", np.nan) for t in tags]

        ax.bar(x - width, r1d,  width, label="1D",    color="#888888", alpha=0.85)
        ax.bar(x,         r_xy, width, label="3D xy", color="#4C9BE8", alpha=0.85)
        ax.bar(x + width, r_xz, width, label="3D xz", color="#E8844C", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=FONT_SIZE * 0.75)
        ax.set_ylabel("R [AU]", fontsize=FONT_SIZE * 0.75)
        ax.tick_params(labelsize=FONT_SIZE * 0.7)
        ax.set_title(format_species_label(mol), fontsize=FONT_SIZE * 0.9)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.4)

        if idx == 0:
            ax.legend(fontsize=FONT_SIZE * 0.65, framealpha=0.7)

    # fig.suptitle(
    #     f"Photodissociation radii — 1D vs 3D  ({chemistry})",
    #     fontsize=FONT_SIZE,
    #     y=1.01,
    # )
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"radii_per_molecule_{chemistry}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved: {fname}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def is_interactive():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare 1D and 3D photodissociation radii per molecule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--radii-file",
        type=str,
        required=True,
        help="Path to the contour_radii.txt written by render_slice_v2.py.",
    )
    parser.add_argument(
        "--chemistry",
        choices=["Crich", "Orich"],
        default="Crich",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save output figures (default: same folder as radii file).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
    )
    if is_interactive():
        args = parser.parse_args([
            "--radii-file", "/fred/oz304/beckers/v10a09_out/output/radii_contours.txt",
            "--chemistry", "Crich",
            "--save-dir", "/fred/oz304/beckers/Chemistry/figures/v10a09_out/Crich",
            "--show",
            # "--dark-mode",
        ])
    else:
        args = parser.parse_args()

    set_plot_style(args.dark_mode)

    radii_file = Path(args.radii_file)
    if not radii_file.exists():
        sys.exit(f"ERROR: radii file not found: {radii_file}")

    save_dir = Path(args.save_dir) if args.save_dir else radii_file.parent

    # ------------------------------------------------------------------
    # Load 3D radii
    # ------------------------------------------------------------------
    radii_3d = load_3d_radii(radii_file)
    molecules_3d = list(radii_3d.keys())

    # ------------------------------------------------------------------
    # Load 1D model and compute radii for the same molecules
    # ------------------------------------------------------------------
    print("Loading 1D model …")
    radius_1d, fracs_1d, coldens, nH2, mloss_label, vinf_label = load_1d_data(
        args.chemistry
    )

    print("Computing 1D photodissociation radii …")
    radii_1d = compute_1d_radii(radius_1d, fracs_1d, molecules_3d)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n--- Radii summary (AU) ---")
    header = f"{'Molecule':>12}  {'1D 0.01':>10}  {'1D 0.50':>10}  "
    header += f"{'3D xy 0.01':>12}  {'3D xy 0.50':>12}  "
    header += f"{'3D xz 0.01':>12}  {'3D xz 0.50':>12}"
    print(header)

    for mol in sorted(set(radii_1d) & set(radii_3d), key=str.lower):
        r1 = radii_1d[mol]
        r3 = radii_3d[mol]
        def fmt(v): return f"{v:10.1f}" if np.isfinite(v) else "         —"
        print(
            f"{mol:>12}  {fmt(r1.get('001', np.nan))}  {fmt(r1.get('050', np.nan))}  "
            f"{fmt(r3.get('R_xy_001', np.nan))}  {fmt(r3.get('R_xy_050', np.nan))}  "
            f"{fmt(r3.get('R_xz_001', np.nan))}  {fmt(r3.get('R_xz_050', np.nan))}"
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating scatter comparison …")
    plot_comparison(radii_1d, radii_3d, save_dir, args.show, args.chemistry)

    print("Generating per-molecule bar chart …")
    plot_radii_per_molecule(radii_1d, radii_3d, save_dir, args.show, args.chemistry)


if __name__ == "__main__":
    main()