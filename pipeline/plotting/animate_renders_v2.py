#!/usr/bin/env python3
"""
animate_renders_v2.py — Unified SPH render animation pipeline.

Modes: single | parent_daughter | phys_overview | mol_grid

Configuration via YAML file or interactive prompts.
  - Terminal sessions: --config is optional; falls back to interactive prompts.
  - SLURM jobs:        --config MUST be an absolute path.  The script fails
                       immediately if the file is missing instead of blocking
                       on stdin, which would hang the job until it times out.

CLI overrides allow fine-grained control from SLURM --export.

Special quantity keywords:
    parents    – render each parent species separately
    daughters  – render each daughter species separately
    molecules  – render all molecules
In 'single' mode this produces one animation per species.
In 'mol_grid' mode the whole list is shown as one grid.

Assembly (run_assemble) uses matplotloom as a context manager, matching the
pattern used in the companion density-trace script.  Each pre-rendered PNG
frame is loaded into a matplotlib figure and passed to loom.save_frame(fig),
so that matplotloom's standard ffmpeg pipeline handles video encoding.
"""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml
from tqdm import tqdm

# Make sure the sibling render module can be imported
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from render_slice_v2 import (  # noqa: E402
    DUMP_DIR,
    PHANTOM_DIR,
    SAVE_DIR_BASE,
    get_all_dumps,
    resolve_dump_pair,
    prepare_render_dataframe,
    render_quantity,
    format_render_axes,
    quantity_label,
    set_plot_style,
    FONT_SIZE,
    plot_parent_daughter_pair,
    plot_phys_overview,
    plot_molecule_grid,
    _read_time_yr,   
    _add_time_label, 
)

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for HPC
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Global DPI — used consistently for both frame saving and assembly so that
# re-wrapping PNGs through matplotlib never changes the pixel dimensions.
# ---------------------------------------------------------------------------
FRAME_DPI = 150

# ---------------------------------------------------------------------------
# Special species groups (mirror render_slice_v2.py)
# ---------------------------------------------------------------------------
PARENTS = [
    "CO", "N2", "CH4", "H2O", "SiC2", "CS", "C2H2",
    "HCN", "SiS", "SiO", "HCl", "C2H4", "NH3", "HCP", "HF", "H2S",
]
DAUGHTERS = [
    "CH2", "CH3", "CH3CN", "CN", "HC3N", "HC5N", "HC7N",
    "C2H", "C4H", "C6H", "SiC", "SiN", "H2CS", "H2CO", "SO", "SO2",
]
MOLECULES = [
    "CO", "CH2", "CH3", "CH4", "HCl", "CH3CN", "SiO", "HCN",
    "CN", "HC3N", "HC5N", "HC7N", "C2H", "C4H", "C6H", "SiC",
    "SiN", "H2CS", "H2CO", "N2", "NH3", "H2S", "HCP", "H2O",
    "C2H2", "CS", "SiC2", "HF", "C2H4", "SiS",
]

SPECIAL_GROUPS = {
    "parents": PARENTS,
    "daughters": DAUGHTERS,
    "molecules": MOLECULES,
}

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def interactive_config() -> dict:
    """Build configuration interactively (terminal sessions only)."""
    config: dict = {}
    print("=== Animation Configuration ===")
    config["mode"] = input(
        "Mode [single/parent_daughter/phys_overview/mol_grid]: "
    ).strip()

    if config["mode"] == "single":
        config["quantity"] = input(
            "Species (name, or 'parents'/'daughters'/'molecules'): "
        ).strip()
    elif config["mode"] == "parent_daughter":
        pair = input("Parent,daughter (e.g., C2H2,C2H): ").split(",")
        if len(pair) != 2:
            raise ValueError("parent_daughter mode needs exactly 2 species.")
        config["parent_daughter_pair"] = [p.strip() for p in pair]
    elif config["mode"] == "mol_grid":
        species = input("Species list (comma-separated, or 'parents'): ")
        config["quantity"] = [s.strip() for s in species.split(",")]
    # phys_overview needs no quantity

    config["plane"] = input("Plane [xy/xz]: ").strip()
    config["chemistry"] = input("Chemistry [Crich/Orich]: ").strip()
    config["dark_mode"] = (
        input("Dark mode? [y/N]: ").strip().lower() in ("y", "yes")
    )
    config["col_dens"] = (
        input("Column-density weighting? [y/N]: ").strip().lower() in ("y", "yes")
    )
    config["dens_weight"] = (
        input("Density weight? [y/N]: ").strip().lower() in ("y", "yes")
    )
    config["cmap"] = input("Colormap (default gist_heat): ").strip() or "gist_heat"
    config["xlim_mode"] = input("xlim mode [fixed/ramp]: ").strip()
    if config["xlim_mode"] == "fixed":
        config["xlim"] = float(input("xlim (AU): ") or 1000)
    else:
        config["xlim_start"] = float(input("xlim start (AU): ") or 100)
        config["xlim_end"] = float(input("xlim end (AU): ") or 2000)
    xsec_str = input("xsec (AU, empty for column integration): ").strip()
    config["xsec"] = float(xsec_str) if xsec_str else None
    config["fps"] = int(input("FPS (default 15): ") or 15)
    return config


def load_config(args) -> dict:
    """Load config from YAML file or interactive prompt, then apply CLI overrides.

    Key behaviour:
      - If --config is given the file MUST exist.  A FileNotFoundError is raised
        immediately with the resolved absolute path, so SLURM jobs fail fast
        instead of hanging on a missing TTY.
      - If --config is omitted the script falls back to interactive prompts,
        which only works in a real terminal session.
    """
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"  (resolved from cwd={Path.cwd()})\n"
                f"  Tip: always pass an absolute path when submitting via SLURM."
            )
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = interactive_config()

    # --- CLI overrides (useful for SLURM --export) ---
    # Only update when the CLI argument was actually supplied; argparse returns
    # None / False for absent optional arguments, so each check is explicit.
    if args.mode:
        config["mode"] = args.mode
    if args.quantity:
        # nargs="+" gives a list; unwrap single-element lists for convenience
        config["quantity"] = (
            args.quantity[0] if len(args.quantity) == 1 else args.quantity
        )
    if args.plane:
        config["plane"] = args.plane
    if args.chemistry:
        config["chemistry"] = args.chemistry
    if args.dark_mode:          # store_true — only True when flag is present
        config["dark_mode"] = True
    if args.col_dens:
        config["col_dens"] = True
    if args.dens_weight:
        config["dens_weight"] = True
    if args.cmap:
        config["cmap"] = args.cmap
    if args.xlim_mode:
        config["xlim_mode"] = args.xlim_mode
    if args.xlim is not None:
        config["xlim"] = args.xlim
    if args.xlim_start is not None:
        config["xlim_start"] = args.xlim_start
    if args.xlim_end is not None:
        config["xlim_end"] = args.xlim_end
    if args.xsec is not None:
        config["xsec"] = args.xsec
    if args.fps:
        config["fps"] = args.fps
    if args.frame_fraction is not None:
        config["frame_fraction"] = args.frame_fraction

    # --- Defaults ---
    config.setdefault("xlim_mode", "fixed")
    config.setdefault("xlim", 1000.0)
    config.setdefault("xlim_start", 100.0)
    config.setdefault("xlim_end", 2000.0)
    config.setdefault("xsec", None)
    config.setdefault("fps", 15)
    config.setdefault("col_dens", False)
    config.setdefault("dens_weight", False)
    config.setdefault("cmap", "gist_heat")
    config.setdefault("chemistry", "Crich")
    config.setdefault("dark_mode", False)
    config.setdefault("dump_dir", str(DUMP_DIR))
    config.setdefault("phantom_dir", str(PHANTOM_DIR))
    config.setdefault("frame_fraction", 1.0)

    # Expand special quantity keywords → lists of actual species names
    q = config.get("quantity")
    if isinstance(q, str) and q.lower() in SPECIAL_GROUPS:
        config["quantity"] = SPECIAL_GROUPS[q.lower()]

    # mol_grid always needs a list
    if config["mode"] == "mol_grid" and not isinstance(config.get("quantity"), list):
        config["quantity"] = [config["quantity"]]

    if config["mode"] == "parent_daughter" and "parent_daughter_pair" not in config:
        raise ValueError(
            "'parent_daughter' mode requires 'parent_daughter_pair' in the config."
        )

    return config


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------
def _quantity_tag(config: dict, quantity_override: str | None = None) -> str:
    if config["mode"] == "phys_overview":
        return "phys"
    if config["mode"] == "parent_daughter":
        pair = config.get("parent_daughter_pair", ["unknown", "unknown"])
        return f"{pair[0]}_{pair[1]}"
    if config["mode"] == "single":
        q = quantity_override if quantity_override else config.get("quantity")
        if isinstance(q, list):
            return "_".join(q[:3]) + ("_etc" if len(q) > 3 else "")
        return str(q)
    if config["mode"] == "mol_grid":
        q_list = config["quantity"]
        return "_".join(q_list[:3]) + ("_etc" if len(q_list) > 3 else "")
    return config["mode"]  # fallback: at least shows what mode was used

def frame_dir(config: dict, quantity_override: str | None = None) -> Path:
    cd_tag = "n" if config["col_dens"] else ""
    xlim_tag = (
        f"fixed{config['xlim']:.0f}AU"
        if config["xlim_mode"] == "fixed"
        else f"ramp{config['xlim_start']:.0f}to{config['xlim_end']:.0f}AU"
    )
    q_tag = _quantity_tag(config, quantity_override)
    if config["mode"] == "phys_overview":
        d = (
            SAVE_DIR_BASE / config["chemistry"] / "animation"
            / f"{q_tag}_{cd_tag}{xlim_tag}" / "frames")
    else:
        d = (
            SAVE_DIR_BASE / config["chemistry"] / "animation"
            / f"{config['plane']}_{cd_tag}{q_tag}_{xlim_tag}" / "frames"
        )
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_path(config: dict, quantity_override: str | None = None) -> Path:
    cd_tag = "n" if config["col_dens"] else ""
    xlim_tag = (
        f"fixed{config['xlim']:.0f}AU"
        if config["xlim_mode"] == "fixed"
        else f"ramp{config['xlim_start']:.0f}to{config['xlim_end']:.0f}AU"
    )
    q_tag = _quantity_tag(config, quantity_override)
    out_dir = SAVE_DIR_BASE / config["chemistry"] / "animation"
    out_dir.mkdir(parents=True, exist_ok=True)
    if config['mode'] == "phys_overview":
        return out_dir / f"{q_tag}_{cd_tag}{xlim_tag}_{config['fps']}fps.mp4"
    else:
        return out_dir / f"{config['plane']}_{cd_tag}{q_tag}_{xlim_tag}_{config['fps']}fps.mp4"


# ---------------------------------------------------------------------------
# Per-frame rendering — all functions use FRAME_DPI consistently
# ---------------------------------------------------------------------------
def render_single_frame(
    dump_index: int,
    dump_dir: Path,
    phantom_dir: Path,
    config: dict,
    xlim: float,
    frame_file: Path,
) -> None:
    phantom_path, dump_path = resolve_dump_pair(dump_dir, phantom_dir, None, dump_index)
    time_yr = _read_time_yr(dump_path)
    quantity = config["quantity"]
    sdf, meta = prepare_render_dataframe(
        phantom_path, dump_path, quantity,
        fraction=1.0, col_dens=config["col_dens"],
    )
    fig, ax = plt.subplots(dpi=FRAME_DPI)
    render_quantity(
        sdf=sdf, quantity=quantity, plane=config["plane"],
        ax=ax, xlim=xlim, xsec=config["xsec"],
        dens_weight=config["dens_weight"], cmap=config["cmap"],
        log_scale=True, interpolate=False, fraction=1.0,
    )
    format_render_axes(
        ax, quantity, config["plane"], meta,
        xlim=xlim, xsec=config["xsec"], plot_single=True,
    )
    mappable = ax.images[0]
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(
        quantity_label(quantity, config["dens_weight"], col_dens=config["col_dens"]),
        fontsize=FONT_SIZE / 2,
    )
    cbar.ax.tick_params(labelsize=FONT_SIZE / 2)
    if time_yr is not None:                      
        _add_time_label(fig, time_yr)
    plt.tight_layout()
    fig.savefig(frame_file, dpi=FRAME_DPI, bbox_inches="tight")
    plt.close(fig)


def render_parent_daughter_frame(
    dump_index: int,
    dump_dir: Path,
    phantom_dir: Path,
    config: dict,
    xlim: float,
    frame_file: Path,
) -> None:
    phantom_path, dump_path = resolve_dump_pair(dump_dir, phantom_dir, None, dump_index)
    time_yr = _read_time_yr(dump_path)
    parent, daughter = config["parent_daughter_pair"]
    plot_parent_daughter_pair(
        phantom_path=phantom_path,
        dump_path=dump_path,
        parent=parent,
        daughter=daughter,
        plane=config["plane"],
        xlim=xlim,
        xsec=config["xsec"],
        dens_weight=config["dens_weight"],
        interpolate=False,
        cmap=config["cmap"],
        save_path=frame_file,
        show=False,
        col_dens=config["col_dens"],
        time_yr=time_yr,
    )


def render_phys_overview_frame(
    dump_index: int,
    dump_dir: Path,
    phantom_dir: Path,
    config: dict,
    xlim: float,
    frame_file: Path,
) -> None:
    phantom_path, dump_path = resolve_dump_pair(dump_dir, phantom_dir, None, dump_index)
    time_yr = _read_time_yr(dump_path)
    plot_phys_overview(
        phantom_path=phantom_path,
        dump_path=dump_path,
        xlim=xlim,
        save_path=frame_file,
        show=False,
        log_scale=True,
        time_yr=time_yr,
    )


def render_mol_grid_frame(
    dump_index: int,
    dump_dir: Path,
    phantom_dir: Path,
    config: dict,
    xlim: float,
    frame_file: Path,
) -> None:
    phantom_path, dump_path = resolve_dump_pair(dump_dir, phantom_dir, None, dump_index)
    time_yr = _read_time_yr(dump_path)
    plot_molecule_grid(
        phantom_path=phantom_path,
        dump_path=dump_path,
        quantities=config["quantity"],
        plane=config["plane"],
        xlim=xlim,
        xsec=config["xsec"],
        dens_weight=config["dens_weight"],
        interpolate=False,
        log_scale=True,
        cmap=config["cmap"],
        save_path=frame_file,
        show=False,
        col_dens=config["col_dens"],
        time_yr=time_yr,
    )


RENDER_FUNCS = {
    "single": render_single_frame,
    "parent_daughter": render_parent_daughter_frame,
    "phys_overview": render_phys_overview_frame,
    "mol_grid": render_mol_grid_frame,
}

# ---------------------------------------------------------------------------
# Frame generation (run as a SLURM array task)
# ---------------------------------------------------------------------------
def run_frames(args) -> None:
    config = load_config(args)
    set_plot_style(config["dark_mode"])

    dump_dir = Path(config["dump_dir"])
    phantom_dir = Path(config["phantom_dir"])
    dumps = get_all_dumps(dump_dir)
    n_dumps = len(dumps)
    if n_dumps == 0:
        print("No dumps found")
        sys.exit(1)

    xlims: list[float] = (
        [config["xlim"]] * n_dumps
        if config["xlim_mode"] == "fixed"
        else np.linspace(config["xlim_start"], config["xlim_end"], n_dumps).tolist()
    )

    # Normalise quantity list; give modes without a quantity a one-shot placeholder
    quantities: list[str] = config.get("quantity", [])
    if not isinstance(quantities, list):
        quantities = [quantities]
    if config["mode"] == "phys_overview":
        quantities = ["dummy"]
    if config["mode"] == "parent_daughter":
        quantities = ["pair"]

    task_id = args.task_id
    n_tasks = args.n_tasks
    frame_fraction = config.get("frame_fraction", 1.0)
    if frame_fraction < 1.0:
        n_keep = max(1, round(n_dumps * frame_fraction))
        selected = sorted(set(
            round(i * (n_dumps - 1) / (n_keep - 1))
            for i in range(n_keep)
        )) if n_keep > 1 else [n_dumps // 2]
    else:
        selected = list(range(n_dumps))

    # Distribute by position in selected list, not by dump index,
    # so tasks get equal work regardless of which indices were picked.
    my_indices = [selected[i] for i in range(len(selected)) if i % n_tasks == task_id]
    print(
        f"Task {task_id}/{n_tasks}: {len(my_indices)} frames "
        f"for {len(quantities)} species/group(s)"
    )

    render_func = RENDER_FUNCS[config["mode"]]

    for q in quantities:
        # deepcopy prevents the per-species quantity override from leaking
        # across iterations when the list contains mutable objects.
        local_config = copy.deepcopy(config)
        if config["mode"] == "single":
            local_config["quantity"] = q

        fdir = frame_dir(
            local_config,
            quantity_override=q if config["mode"] == "single" else None,
        )

        for dump_index in my_indices:
            frame_file = fdir / f"frame_{dump_index:05d}.png"
            sentinel = fdir / f"FAILED_{dump_index:05d}.txt"

            # # Skip already-rendered frames; size check catches zero-byte/truncated files
            # # that a previously crashed task may have left behind.
            # if frame_file.exists() and frame_file.stat().st_size > 1024:
            #     print(f"[{q} frame {dump_index:05d}] already exists, skipping")
            #     continue

            # Clear any leftover failure sentinel before retrying
            if sentinel.exists():
                sentinel.unlink()

            try:
                render_func(
                    dump_index=dump_index,
                    dump_dir=dump_dir,
                    phantom_dir=phantom_dir,
                    config=local_config,
                    xlim=xlims[dump_index],
                    frame_file=frame_file,
                )
                print(f"[{q} frame {dump_index:05d}] saved → {frame_file}")
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[{q} frame {dump_index:05d}] ERROR: {exc}\n{tb}")
                # Write a sentinel so the assembly step can warn about missing frames
                # and so re-runs can skip past frames that rendered successfully.
                sentinel.write_text(tb)


# ---------------------------------------------------------------------------
# Assembly — stitch pre-rendered PNG frames into a video via matplotloom
#
# Uses the same context-manager + save_frame(fig) pattern as the companion
# density-trace script.  Each cached PNG is loaded into a matplotlib figure
# and handed to loom.save_frame(), so matplotloom's ffmpeg pipeline handles
# all encoding.  savefig_kwargs use pad_inches=0 (not bbox_inches="tight")
# to guarantee identical pixel dimensions across every frame.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Assembly — call ffmpeg directly on the pre-rendered PNG frames.
# Uses the concat demuxer so missing/non-contiguous frames are handled
# cleanly without any temp directories.
# ---------------------------------------------------------------------------
def run_assemble(args) -> None:
    if getattr(args, "all_configs", False):
        run_assemble_all(args)
        return
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        print(
            "ffmpeg binary not found on PATH.\n"
            "On the cluster try:  module load ffmpeg\n"
            "Or install via:      conda install -c conda-forge ffmpeg"
        )
        sys.exit(1)

    config = load_config(args)
    set_plot_style(config["dark_mode"])

    quantities: list[str] = config.get("quantity", [])
    if not isinstance(quantities, list):
        quantities = [quantities]
    if config["mode"] == "phys_overview":
        quantities = ["dummy"]
    if config["mode"] == "parent_daughter":
        quantities = ["pair"]

    dumps = get_all_dumps(Path(config["dump_dir"]))
    n_dumps = len(dumps)

    for q in quantities:
        local_config = copy.deepcopy(config)
        if config["mode"] == "single":
            local_config["quantity"] = q

        fdir = frame_dir(
            local_config,
            quantity_override=q if config["mode"] == "single" else None,
        )
        out_path = video_path(
            local_config,
            quantity_override=q if config["mode"] == "single" else None,
        )

        # Report frames that errored during rendering
        failed_indices = sorted(
            int(p.stem.split("_")[1])
            for p in fdir.glob("FAILED_?????.txt")
        )
        if failed_indices:
            print(
                f"[{q}] WARNING: {len(failed_indices)} frame(s) failed during rendering "
                f"(indices {failed_indices[:10]}"
                f"{'...' if len(failed_indices) > 10 else ''}).  "
                f"Re-run the frames step to retry them."
            )

        # Collect valid (non-empty) frame files in dump order
        frame_files: list[Path] = []
        missing: list[int] = []
        for i in range(n_dumps):
            p = fdir / f"frame_{i:05d}.png"
            if p.exists() and p.stat().st_size > 1024:
                frame_files.append(p)
            else:
                missing.append(i)

        if missing:
            print(
                f"[{q}] WARNING: {len(missing)} frame(s) missing or corrupt "
                f"(indices {missing[:10]}{'...' if len(missing) > 10 else ''})"
            )

        if not frame_files:
            print(f"[{q}] No valid frames found – skipping")
            continue

        print(f"[{q}] Assembling {len(frame_files)} frames → {out_path}")

        # Write a concat list next to the frames (no /tmp involved).
        # The final entry is repeated without a duration — required by ffmpeg's
        # concat demuxer to flush the last frame correctly.
        duration = 1.0 / config["fps"]
        concat_file = fdir / "_concat.txt"
        with open(concat_file, "w") as cf:
            for frame_file in frame_files:
                cf.write(f"file '{frame_file.resolve()}'\n")
                cf.write(f"duration {duration:.6f}\n")
            cf.write(f"file '{frame_files[-1].resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2"
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",   # broadest player compatibility
            "-crf", "18",            # visually lossless; raise to 23 to shrink file
            str(out_path),
        ]
        print(f"[{q}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        concat_file.unlink()

        if result.returncode != 0:
            print(f"[{q}] ffmpeg failed (return code {result.returncode})")
        else:
            print(f"[{q}] Done → {out_path}")

def run_assemble_all(args) -> None:
    """Scan the animation directory and assemble every frame set found."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        print(
            "ffmpeg not found on PATH.\n"
            "Try: module load ffmpeg  or  conda install -c conda-forge ffmpeg"
        )
        sys.exit(1)

    fps        = args.fps or 15
    chemistry  = args.chemistry or "Crich"
    anim_dir   = SAVE_DIR_BASE / chemistry / "animation"

    if not anim_dir.exists():
        print(f"Animation directory not found: {anim_dir}")
        sys.exit(1)

    frame_dirs = sorted(anim_dir.glob("*/frames"))
    if not frame_dirs:
        print(f"No frame directories found under {anim_dir}")
        return

    print(f"Found {len(frame_dirs)} frame set(s) under {anim_dir}\n")

    for fdir in frame_dirs:
        tag      = fdir.parent.name          # e.g. xy_CO_fixed1000AU_p10
        out_path = anim_dir / f"{tag}_{fps}fps.mp4"

        failed = sorted(fdir.glob("FAILED_?????.txt"))
        if failed:
            print(f"[{tag}] WARNING: {len(failed)} failed frame(s)")

        frame_files = sorted(
            (p for p in fdir.glob("frame_?????.png") if p.stat().st_size > 1024),
            key=lambda p: int(p.stem.split("_")[1]),
        )

        if not frame_files:
            print(f"[{tag}] No valid frames — skipping")
            continue

        print(f"[{tag}] {len(frame_files)} frames → {out_path}")

        duration    = 1.0 / fps
        concat_file = fdir / "_concat.txt"
        with open(concat_file, "w") as cf:
            for f in frame_files:
                cf.write(f"file '{f.resolve()}'\n")
                cf.write(f"duration {duration:.6f}\n")
            cf.write(f"file '{frame_files[-1].resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(out_path),
        ]
        result = subprocess.run(cmd)
        concat_file.unlink()

        if result.returncode != 0:
            print(f"[{tag}] ffmpeg failed (return code {result.returncode})")
        else:
            print(f"[{tag}] Done\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified SPH animation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_overrides(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            help="Path to YAML config file.  Use an absolute path for SLURM jobs.",
        )
        p.add_argument(
            "--mode",
            choices=["single", "parent_daughter", "phys_overview", "mol_grid"],
        )
        p.add_argument("--quantity", nargs="+")
        p.add_argument("--plane", choices=["xy", "xz"])
        p.add_argument("--chemistry", choices=["Crich", "Orich"])
        # store_true flags must NOT receive a value on the command line.
        # The SLURM scripts use ${VAR:+--flag} to emit them only when set.
        p.add_argument("--dark-mode", action="store_true", default=False)
        p.add_argument("--col-dens", action="store_true", default=False)
        p.add_argument("--dens-weight", action="store_true", default=False)
        p.add_argument("--cmap")
        p.add_argument("--xlim-mode", choices=["fixed", "ramp"])
        p.add_argument("--xlim", type=float)
        p.add_argument("--xlim-start", type=float)
        p.add_argument("--xlim-end", type=float)
        p.add_argument("--xsec", type=float)
        p.add_argument("--fps", type=int)
        p.add_argument("--frame-fraction", type=float,
               help="Fraction of frames to render, evenly spaced (e.g. 0.1 for 10%%)")

    p_frames = sub.add_parser("frames", help="Render a batch of frames (SLURM array task)")
    add_overrides(p_frames)
    p_frames.add_argument("--task-id", type=int, default=0)
    p_frames.add_argument("--n-tasks", type=int, default=1)

    p_assemble = sub.add_parser(
        "assemble",
        help="Stitch pre-rendered frames into MP4 via matplotloom",
    )
    p_assemble.add_argument(
    "--all", dest="all_configs", action="store_true",
    help="Scan the animation directory and assemble all available frame sets.",
    )
    add_overrides(p_assemble)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "frames":
        run_frames(args)
    elif args.command == "assemble":
        run_assemble(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()