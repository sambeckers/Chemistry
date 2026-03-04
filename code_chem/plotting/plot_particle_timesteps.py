#!/usr/bin/env python3
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import matplotlib.pyplot as plt


DIRECTORY = Path("/fred/oz304/beckers/Chemistry/traces/wind_v10/trace_output_with_av")
PATTERN = "*.phys"
COMMENT_PREFIX = "#"
RECURSIVE = False
OUTPUT_FILE = Path("/fred/oz304/beckers/Chemistry/figures/particle_id_vs_timesteps.png")
FIGSIZE = (11, 5)
DPI = 180


def count_data_rows(file_path: Path, comment_prefix: str = "#") -> int:
    rows = 0
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if comment_prefix and stripped.startswith(comment_prefix):
                continue
            rows += 1
    return rows


def collect_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in root.rglob(pattern) if path.is_file())
    return sorted(path for path in root.glob(pattern) if path.is_file())


def particle_id_from_file(file_path: Path) -> int | None:
    try:
        return int(file_path.stem)
    except ValueError:
        return None


def main() -> None:
    directory = DIRECTORY.expanduser()

    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    files = collect_files(directory, PATTERN, RECURSIVE)
    particle_ids: list[int] = []
    timesteps: list[int] = []

    for file_path in files:
        particle_id = particle_id_from_file(file_path)
        if particle_id is None:
            continue
        row_count = count_data_rows(file_path, comment_prefix=COMMENT_PREFIX)
        particle_ids.append(particle_id)
        timesteps.append(row_count)

    if not particle_ids:
        raise SystemExit("No valid numeric particle IDs were found.")

    pairs = sorted(zip(particle_ids, timesteps), key=lambda item: item[0])
    particle_ids = [item[0] for item in pairs]
    timesteps = [item[1] for item in pairs]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(particle_ids, timesteps, linewidth=1.0)
    ax.set_xlim(left=particle_ids[0])
    ax.set_xlabel("Particle ID")
    ax.set_ylabel("Number of timesteps")
    # ax.set_title("Particle ID vs Number of Timesteps")
    ax.grid(True, alpha=0.25)

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    bottom_ticks = ax.get_xticks()
    cumulative_counts = [bisect_right(particle_ids, tick) for tick in bottom_ticks]
    top_ax.set_xticks(bottom_ticks)
    top_ax.set_xticklabels([str(count) for count in cumulative_counts])
    top_ax.set_xlabel("Cumulative particle count (<= Particle ID)")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=DPI)

    print(f"Directory: {directory}")
    print(f"Pattern: {PATTERN}")
    print(f"Points plotted: {len(particle_ids)}")
    print(f"Output plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
