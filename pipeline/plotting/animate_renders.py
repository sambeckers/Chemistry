#!/usr/bin/env python3
"""
animate_renders.py

Generate per-frame PNG renders (embarrassingly parallel via SLURM) and
assemble them into an MP4 animation with matplotloom.

Two modes
---------
frames  -- render one batch of dump-frames (called by the SLURM array job)
assemble -- stitch the already-rendered frames into a video

Limit modes
-----------
fixed   -- xlim is constant (e.g. 1000 AU) throughout the animation
ramp    -- xlim grows linearly from XLIM_START to XLIM_END over all dumps

Usage examples
--------------
# 1. Submit frame jobs
sbatch --export=ALL,MODE=frames,QUANTITY=CO,PLANE=xz,XLIM_MODE=fixed,XLIM=1000 \
       animate_renders.slurm

# 2. Assemble (run locally or in a short interactive session)
python animate_renders.py assemble \
    --quantity CO --plane xz \
    --xlim-mode fixed --xlim 1000 \
    --fps 15

# Or with a ramping window:
python animate_renders.py assemble \
    --quantity CO --plane xz \
    --xlim-mode ramp --xlim-start 100 --xlim-end 2000 \
    --fps 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: make render_slice_v2 importable regardless of working directory
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Import everything we need from the sibling script
from render_slice_v2 import (          # noqa: E402
    DUMP_DIR,
    PHANTOM_DIR,
    SAVE_DIR_BASE,
    get_all_dumps,
    resolve_dump_pair,
    prepare_render_dataframe,
    render_quantity,
    format_render_axes,
    quantity_label,
    set_plot_style,          # re-exported from plot_utils via render_slice_v2
    FONT_SIZE,
)

# If set_plot_style lives in plot_utils rather than render_slice_v2 you may
# need to import it directly:
try:
    from render_slice_v2 import set_plot_style  # noqa: F811
except ImportError:
    from code_chem.plotting.plot_utils import set_plot_style

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for HPC nodes
import matplotlib.pyplot as plt

# ===========================================================================
# Defaults — override via CLI or SLURM --export
# ===========================================================================
DEFAULT_FPS        = 15
DEFAULT_XLIM_MODE  = "fixed"       # "fixed" | "ramp"
DEFAULT_XLIM       = 1000.0        # AU  (used when xlim-mode=fixed)
DEFAULT_XLIM_START = 100.0         # AU  (ramp start)
DEFAULT_XLIM_END   = 2000.0        # AU  (ramp end)
DEFAULT_CMAP       = "gist_heat"
DEFAULT_CHEMISTRY  = "Crich"

# ===========================================================================
# Frame output directory
# ===========================================================================
def frame_dir(
    quantity: str,
    plane: str,
    xlim_mode: str,
    xlim_val: float | None,
    xlim_start: float | None,
    xlim_end: float | None,
    chemistry: str,
    col_dens: bool,
) -> Path:
    cd_tag = "n" if col_dens else ""
    if xlim_mode == "fixed":
        xlim_tag = f"fixed{xlim_val:.0f}AU"
    else:
        xlim_tag = f"ramp{xlim_start:.0f}to{xlim_end:.0f}AU"

    d = (
        SAVE_DIR_BASE
        / chemistry
        / "animation"
        / f"{plane}_{cd_tag}{quantity}_{xlim_tag}"
        / "frames"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_path(
    quantity: str,
    plane: str,
    xlim_mode: str,
    xlim_val: float | None,
    xlim_start: float | None,
    xlim_end: float | None,
    chemistry: str,
    col_dens: bool,
    fps: int,
) -> Path:
    cd_tag = "n" if col_dens else ""
    if xlim_mode == "fixed":
        xlim_tag = f"fixed{xlim_val:.0f}AU"
    else:
        xlim_tag = f"ramp{xlim_start:.0f}to{xlim_end:.0f}AU"

    out_dir = SAVE_DIR_BASE / chemistry / "animation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{plane}_{cd_tag}{quantity}_{xlim_tag}_{fps}fps.mp4"


# ===========================================================================
# Compute per-dump xlim
# ===========================================================================
def compute_xlims(
    n_dumps: int,
    xlim_mode: str,
    xlim_val: float | None,
    xlim_start: float | None,
    xlim_end: float | None,
) -> list[float]:
    """Return a list of xlim values, one per dump."""
    if xlim_mode == "fixed":
        return [xlim_val] * n_dumps
    # ramp: linear interpolation
    return list(np.linspace(xlim_start, xlim_end, n_dumps))


# ===========================================================================
# Render a single frame
# ===========================================================================
def render_frame(
    dump_index: int,
    quantity: str,
    plane: str,
    xlim: float,
    xsec: float | None,
    dens_weight: bool,
    cmap: str,
    col_dens: bool,
    frame_file: Path,
    dump_dir: Path = DUMP_DIR,
    phantom_dir: Path = PHANTOM_DIR,
):
    """Render one dump as a PNG frame and save it to *frame_file*."""
    phantom_path, dump_path = resolve_dump_pair(
        dump_dir, phantom_dir, None, dump_index
    )

    sdf, meta = prepare_render_dataframe(
        phantom_path,
        dump_path,
        quantity,
        fraction=1.0,
        col_dens=col_dens,
    )

    fig, ax = plt.subplots(dpi=300)

    render_quantity(
        sdf=sdf,
        quantity=quantity,
        plane=plane,
        ax=ax,
        xlim=xlim,
        xsec=xsec,
        dens_weight=dens_weight,
        cmap=cmap,
        log_scale=True,
        interpolate=False,
        fraction=1.0,
    )

    format_render_axes(ax, quantity, plane, meta, xlim=xlim, xsec=xsec, plot_single=True)

    mappable = ax.images[0]
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(quantity_label(quantity, dens_weight, col_dens=col_dens), fontsize=FONT_SIZE / 2)
    cbar.ax.tick_params(labelsize=FONT_SIZE / 2)

    plt.tight_layout()
    fig.savefig(frame_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[frame {dump_index:05d}] saved → {frame_file}")


# ===========================================================================
# MODE: frames  (called by the SLURM array task)
# ===========================================================================
def run_frames(args):
    """Render the subset of frames assigned to this SLURM task."""
    dumps = get_all_dumps(Path(args.dump_dir))
    n_dumps = len(dumps)

    if n_dumps == 0:
        print("No dumps found — check --dump-dir")
        sys.exit(1)

    xlims = compute_xlims(
        n_dumps,
        args.xlim_mode,
        args.xlim,
        args.xlim_start,
        args.xlim_end,
    )

    fdir = frame_dir(
        args.quantity, args.plane,
        args.xlim_mode, args.xlim, args.xlim_start, args.xlim_end,
        args.chemistry, args.col_dens,
    )

    # Partition dump indices among SLURM tasks
    task_id = args.task_id      # 0-based
    n_tasks = args.n_tasks

    my_indices = [i for i in range(n_dumps) if i % n_tasks == task_id]
    print(
        f"Task {task_id}/{n_tasks}: rendering {len(my_indices)} "
        f"of {n_dumps} dumps"
    )

    set_plot_style(dark_mode=False)

    for dump_index in my_indices:
        frame_file = fdir / f"frame_{dump_index:05d}.png"
        if frame_file.exists():
            print(f"[frame {dump_index:05d}] already exists, skipping")
            continue

        try:
            render_frame(
                dump_index=dump_index,
                quantity=args.quantity,
                plane=args.plane,
                xlim=xlims[dump_index],
                xsec=args.xsec,
                dens_weight=args.dens_weight,
                cmap=args.cmap,
                col_dens=args.col_dens,
                frame_file=frame_file,
                dump_dir=Path(args.dump_dir),
                phantom_dir=Path(args.phantom_dir),
            )
        except Exception as exc:
            print(f"[frame {dump_index:05d}] ERROR: {exc}")


# ===========================================================================
# MODE: assemble  (run once after all frames are ready)
# ===========================================================================
def run_assemble(args):
    """Stitch frames into an MP4 with matplotloom."""
    try:
        import matplotloom
    except ImportError:
        print(
            "matplotloom is not installed.\n"
            "Install it with:  pip install matplotloom"
        )
        sys.exit(1)

    dumps = get_all_dumps(Path(args.dump_dir))
    n_dumps = len(dumps)

    fdir = frame_dir(
        args.quantity, args.plane,
        args.xlim_mode, args.xlim, args.xlim_start, args.xlim_end,
        args.chemistry, args.col_dens,
    )

    # Collect frames in order, warn about missing ones
    frame_files = []
    missing = []
    for i in range(n_dumps):
        p = fdir / f"frame_{i:05d}.png"
        if p.exists():
            frame_files.append(p)
        else:
            missing.append(i)

    if missing:
        print(
            f"WARNING: {len(missing)} frame(s) missing "
            f"(indices {missing[:10]}{'...' if len(missing) > 10 else ''}). "
            "These will be skipped in the animation."
        )

    if not frame_files:
        print("No frames found — run 'frames' mode first.")
        sys.exit(1)

    out_path = video_path(
        args.quantity, args.plane,
        args.xlim_mode, args.xlim, args.xlim_start, args.xlim_end,
        args.chemistry, args.col_dens,
        args.fps,
    )

    print(f"Assembling {len(frame_files)} frames → {out_path}")
    print(f"FPS: {args.fps}")

    # matplotloom.Loom assembles a list of image files into a video
    loom = matplotloom.Loom(
        str(out_path),
        fps=args.fps,
        overwrite=True,
    )
    for f in frame_files:
        loom.add_frame(str(f))
    loom.close()

    print(f"Done → {out_path}")


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Animate SPH renders (frame generation + assembly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- shared arguments -----------------------------------------------
    def add_shared(p):
        p.add_argument("--quantity",     default="density")
        p.add_argument("--plane",        choices=["xy", "xz"], default="xy")
        p.add_argument("--xsec",         type=float,  default=None)
        p.add_argument("--dens-weight",  action="store_true", default=False)
        p.add_argument("--col-dens",     action="store_true", default=False)
        p.add_argument("--cmap",         default=DEFAULT_CMAP)
        p.add_argument("--chemistry",    choices=["Crich", "Orich"],
                       default=DEFAULT_CHEMISTRY)
        p.add_argument("--dump-dir",     default=str(DUMP_DIR))
        p.add_argument("--phantom-dir",  default=str(PHANTOM_DIR))

        # xlim mode
        p.add_argument(
            "--xlim-mode",
            choices=["fixed", "ramp"],
            default=DEFAULT_XLIM_MODE,
            help=(
                "fixed: constant xlim for all frames; "
                "ramp: linearly increase xlim from --xlim-start to --xlim-end"
            ),
        )
        p.add_argument(
            "--xlim",
            type=float,
            default=DEFAULT_XLIM,
            help="Constant xlim in AU (used when --xlim-mode=fixed)",
        )
        p.add_argument(
            "--xlim-start",
            type=float,
            default=DEFAULT_XLIM_START,
            help="Starting xlim in AU (used when --xlim-mode=ramp)",
        )
        p.add_argument(
            "--xlim-end",
            type=float,
            default=DEFAULT_XLIM_END,
            help="Ending xlim in AU (used when --xlim-mode=ramp)",
        )

    # ---- frames sub-command ---------------------------------------------
    p_frames = sub.add_parser("frames", help="Render a batch of frames (SLURM task)")
    add_shared(p_frames)
    p_frames.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="SLURM_ARRAY_TASK_ID (0-based)",
    )
    p_frames.add_argument(
        "--n-tasks",
        type=int,
        default=1,
        help="Total number of SLURM array tasks",
    )

    # ---- assemble sub-command -------------------------------------------
    p_assemble = sub.add_parser("assemble", help="Stitch frames into an MP4")
    add_shared(p_assemble)
    p_assemble.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second (default: {DEFAULT_FPS})",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "frames":
        run_frames(args)
    elif args.mode == "assemble":
        run_assemble(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()