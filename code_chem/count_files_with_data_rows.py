#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_PATH


DIRECTORY = BASE_PATH / "traces/wind_v10/trace_output_with_av"
PARTICLE_IDS_FILE = BASE_PATH / "traces/wind_v10/particle_IDs.txt"
PATTERN = "*.phys"
MIN_ROWS = 3
COMMENT_PREFIX = "#"
RECURSIVE = False


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
    pattern = PATTERN
    min_rows = MIN_ROWS
    comment_prefix = COMMENT_PREFIX
    recursive = RECURSIVE

    if directory is None or not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")
    if min_rows < 0:
        raise SystemExit("--min-rows must be >= 0")

    # Load the reference set of valid particle IDs
    valid_particle_ids: set[int] | None = None
    if PARTICLE_IDS_FILE.exists():
        with PARTICLE_IDS_FILE.open() as f:
            valid_particle_ids = {int(line.strip()) for line in f if line.strip()}
    else:
        print(f"Warning: particle_IDs file not found: {PARTICLE_IDS_FILE}")

    files = collect_files(directory, pattern, recursive)
    qualifying = 0
    highest_id_matching_file: Path | None = None
    highest_id_matching_rows: int | None = None
    highest_particle_id: int | None = None
    invalid_particle_ids: list[int] = []

    for file_path in files:
        row_count = count_data_rows(file_path, comment_prefix=comment_prefix)
        particle_id = particle_id_from_file(file_path)
        if row_count > min_rows:
            qualifying += 1
            if particle_id is not None and (
                highest_particle_id is None or particle_id > highest_particle_id
            ):
                highest_particle_id = particle_id
                highest_id_matching_file = file_path
                highest_id_matching_rows = row_count
        if valid_particle_ids is not None and particle_id is not None:
            if particle_id not in valid_particle_ids:
                invalid_particle_ids.append(particle_id)

    print(f"Directory: {directory}")
    print(f"Pattern: {pattern}")
    print(f"Total files scanned: {len(files)}")
    print(f"Files with data rows > {min_rows}: {qualifying}")
    if highest_id_matching_file is None:
        print("Highest-ID matching file: none")
    else:
        print(
            f"Highest-ID matching file: {highest_id_matching_file} "
            f"(particle_id: {highest_particle_id}, rows: {highest_id_matching_rows})"
        )
    if valid_particle_ids is not None:
        # print(f"Particle IDs file: {PARTICLE_IDS_FILE}")
        print(f"Invalid particle IDs (not in particle_IDs.txt): {len(invalid_particle_ids)}")
        if invalid_particle_ids:
            print(f"  IDs: {sorted(invalid_particle_ids)}")


if __name__ == "__main__":
    main()
