#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


DIRECTORY = Path("/fred/oz304/beckers/Chemistry/traces/wind_v10/trace_output_with_av")
PATTERN = "*.phys"
MIN_ROWS = 1200
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

    files = collect_files(directory, pattern, recursive)
    qualifying = 0
    highest_id_matching_file: Path | None = None
    highest_id_matching_rows: int | None = None
    highest_particle_id: int | None = None

    for file_path in files:
        row_count = count_data_rows(file_path, comment_prefix=comment_prefix)
        if row_count > min_rows:
            qualifying += 1
            particle_id = particle_id_from_file(file_path)
            if particle_id is not None and (
                highest_particle_id is None or particle_id > highest_particle_id
            ):
                highest_particle_id = particle_id
                highest_id_matching_file = file_path
                highest_id_matching_rows = row_count

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


if __name__ == "__main__":
    main()
