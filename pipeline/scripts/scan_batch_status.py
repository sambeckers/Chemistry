#!/usr/bin/env python3
"""
scan_batch_status.py  —  Scan SLURM log folders for batch job outcomes.

For each batch_<JOBID>_<TASKID>.out file the script reads only the last
~60 lines (cheap tail) to find:

  1. The SLURM job-report line:  Job Report: NNNNNN (COMPLETED|FAILED)
  2. The pipeline status line:   BATCH_STATUS|batch=N|status=ok|...
  3. The batch wall-time line:   TIMING|batch=N|stage=batch_total|seconds=...

A batch is classified as:
  COMPLETED  — SLURM says COMPLETED  **and**  pipeline emitted status=ok
  FAILED     — SLURM says FAILED, or SLURM says COMPLETED but pipeline
               did not emit status=ok (e.g. killed mid-run)
  RUNNING    — no job-report block found yet

The absolute batch index is:  BATCH_INDEX_OFFSET + SLURM_ARRAY_TASK_ID
The offset is read from the folder's workflow_submit_metadata.txt
(written by submit_batch.sh), falling back to the BATCH_INDEX_OFFSET
value found inside the .out file itself.

Output
------
  batch_status.txt   — one line per batch:  "<index> <STATUS>"
  (stdout)           — human-readable summary, including average/min/max
                        wall-clock time for COMPLETED batches

Usage
-----
  python scan_batch_status.py --log-root /path/to/scratch/logs \\
                               --output   batch_status.txt \\
                               [--total   10000]

  # Scan specific folders (glob patterns accepted):
  python scan_batch_status.py --folders logs_*/  --output batch_status.txt
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Matches:  Job Report: 11557082 (COMPLETED)
#           Job Report: 11608383 (FAILED)
RE_JOB_REPORT = re.compile(
    r"Job Report:\s+\d+\s+\((?P<slurm_status>COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY)\)",
    re.IGNORECASE,
)

# Matches: BATCH_STATUS|batch=42|status=ok|...
RE_BATCH_STATUS_OK = re.compile(r"BATCH_STATUS\|batch=\d+\|status=ok\b")

# Matches: BATCH_INDEX_OFFSET=2048  (exported env var echoed in some logs)
# Also matches the BATCH_TASK line: BATCH_TASK|batch=42|phase=start|...
RE_BATCH_INDEX = re.compile(r"BATCH_TASK\|batch=(?P<idx>\d+)\|phase=start")

# Matches:  TIMING|batch=4|stage=batch_total|seconds=51267.590055
RE_BATCH_TOTAL_TIME = re.compile(
    r"TIMING\|batch=\d+\|stage=batch_total\|seconds=(?P<sec>[\d.]+)"
)

# Matches filename:  batch_<JOBID>_<TASKID>.out
RE_OUT_FILENAME = re.compile(r"^batch_\d+_(?P<task_id>\d+)\.out$")

# Matches workflow_submit_metadata line: chunk_count=5 / batch_count=10000
RE_META_OFFSET = re.compile(r"^#?\s*chunk_offset\s*=\s*(?P<val>\d+)", re.IGNORECASE)

TAIL_LINES = 80   # number of lines to read from the end of each .out file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tail_lines(path: Path, n: int = TAIL_LINES) -> list[str]:
    """Return up to *n* lines from the end of *path* without reading the whole file."""
    # For files up to a few MB a simple seek-from-end is fastest on Lustre.
    chunk = 4096 * n      # generous upper bound
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, size - chunk))
            raw = fh.read()
    except OSError:
        return []
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return lines[-n:]


def head_lines(path: Path, n: int = 5) -> list[str]:
    """Return first *n* lines of *path* (to extract the BATCH_TASK start line)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return [fh.readline().rstrip() for _ in range(n)]
    except OSError:
        return []


def parse_batch_index_from_head(path: Path) -> int | None:
    """Extract absolute batch index from the BATCH_TASK|batch=N|phase=start line."""
    for line in head_lines(path, n=10):
        m = RE_BATCH_INDEX.search(line)
        if m:
            return int(m.group("idx"))
    return None


def classify_out_file(path: Path) -> tuple[str, int | None, float | None]:
    """
    Returns (status, batch_index_or_None, batch_total_seconds_or_None).

    status is one of: 'COMPLETED', 'FAILED', 'RUNNING'
    """
    last = tail_lines(path)
    text = "\n".join(last)

    pipeline_ok = bool(RE_BATCH_STATUS_OK.search(text))

    slurm_match = RE_JOB_REPORT.search(text)
    slurm_status = slurm_match.group("slurm_status").upper() if slurm_match else None

    if slurm_status == "COMPLETED" and pipeline_ok:
        status = "COMPLETED"
    elif slurm_status is not None:
        # SLURM finished but pipeline didn't confirm ok  →  treat as FAILED
        status = "FAILED"
    else:
        # Job-report block not present yet  →  still running (or pending)
        status = "RUNNING"

    batch_idx = parse_batch_index_from_head(path)

    time_match = RE_BATCH_TOTAL_TIME.search(text)
    batch_seconds = float(time_match.group("sec")) if time_match else None

    return status, batch_idx, batch_seconds


def read_chunk_offset_from_metadata(folder: Path) -> int | None:
    """Read chunk_offset from workflow_submit_metadata.txt if present."""
    meta = folder / "workflow_submit_metadata.txt"
    if not meta.is_file():
        return None
    try:
        for line in meta.read_text(encoding="utf-8").splitlines():
            m = RE_META_OFFSET.match(line)
            if m:
                return int(m.group("val"))
    except OSError:
        pass
    return None


def scan_folder(folder: Path) -> tuple[dict[int, str], dict[int, float]]:
    """
    Scan one log folder and return:
      ({absolute_batch_index: status}, {absolute_batch_index: batch_total_seconds})

    The timings dict only contains entries for which a batch_total TIMING
    line was found (regardless of final status).
    """
    results: dict[int, str] = {}
    timings: dict[int, float] = {}

    chunk_offset = read_chunk_offset_from_metadata(folder)

    out_files = sorted(folder.glob("batch_*.out"))
    if not out_files:
        return results, timings

    for out_path in out_files:
        fn_match = RE_OUT_FILENAME.match(out_path.name)
        if not fn_match:
            continue
        task_id = int(fn_match.group("task_id"))

        status, batch_idx_from_log, batch_seconds = classify_out_file(out_path)

        # Resolve absolute batch index in priority order:
        #   1. Parsed from the BATCH_TASK start line inside the file (most reliable)
        #   2. chunk_offset (from metadata) + task_id
        if batch_idx_from_log is not None:
            abs_idx = batch_idx_from_log
        elif chunk_offset is not None:
            abs_idx = chunk_offset + task_id
        else:
            # Fall back: task_id alone (correct only for the first chunk)
            abs_idx = task_id

        # If we see the same index twice (shouldn't happen, but guard anyway),
        # COMPLETED wins over FAILED, FAILED wins over RUNNING.
        priority = {"COMPLETED": 2, "FAILED": 1, "RUNNING": 0}
        if abs_idx not in results or priority[status] > priority[results[abs_idx]]:
            results[abs_idx] = status
            if batch_seconds is not None:
                timings[abs_idx] = batch_seconds
        elif abs_idx not in timings and batch_seconds is not None:
            timings[abs_idx] = batch_seconds

    return results, timings


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format seconds as e.g. '14h 16m 08s' for readability."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_BATCH_FOLDER_RE = re.compile(r"^logs_\d+_")

def discover_log_folders(log_root: Path) -> list[Path]:
    """Find all logs_<JOBID>_<DATETIME> subdirectories under log_root.

    Only matches folders whose name is  logs_<digits>_<anything>, so
    tracing_logs_* and other non-batch folders are automatically skipped.
    """
    folders = []
    for entry in sorted(log_root.iterdir()):
        if entry.is_dir() and _BATCH_FOLDER_RE.match(entry.name):
            folders.append(entry)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan SLURM batch log folders and write a status file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Accept log root as a plain positional OR via --log-root flag so both work:
    #   python scan_batch_status.py /fred/.../logs
    #   python scan_batch_status.py --log-root /fred/.../logs
    parser.add_argument(
        "log_root_pos",
        metavar="LOG_ROOT",
        nargs="?",
        default=None,
        help="Root logs directory (positional shortcut, same as --log-root).",
    )
    parser.add_argument(
        "--log-root",
        metavar="DIR",
        default=None,
        help="Root logs directory. All logs_<JOBID>_* subdirs will be scanned.",
    )
    parser.add_argument(
        "--folders",
        metavar="DIR",
        nargs="+",
        default=None,
        help="Explicit list of log folder paths to scan (alternative to --log-root).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default="batch_status.txt",
        help="Path to write the status file (default: batch_status.txt).",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        metavar="N",
        help="Expected total number of batches (for summary denominator).",
    )
    args = parser.parse_args()

    # Resolve which mode we're in
    log_root_str = args.log_root or args.log_root_pos
    if not log_root_str and not args.folders:
        parser.error("Provide a LOG_ROOT path (positional or --log-root) or use --folders.")

    # Resolve folders
    if log_root_str:
        log_root = Path(log_root_str).resolve()
        if not log_root.is_dir():
            print(f"Error: log root not found: {log_root}", file=sys.stderr)
            sys.exit(1)
        folders = discover_log_folders(log_root)
        if not folders:
            print(f"No logs_<JOBID>_* subdirectories found under {log_root}", file=sys.stderr)
            sys.exit(1)
    else:
        folders = [Path(f).resolve() for f in args.folders]

    print(f"Scanning {len(folders)} log folder(s):")
    for f in folders:
        print(f"  {f}")
    print()

    # Aggregate across all folders
    combined: dict[int, str] = {}
    combined_timings: dict[int, float] = {}
    priority = {"COMPLETED": 2, "FAILED": 1, "RUNNING": 0}
    folder_counts: dict[str, dict[str, int]] = {}

    for folder in folders:
        folder_results, folder_timings = scan_folder(folder)
        folder_counts[folder.name] = defaultdict(int)
        for idx, status in folder_results.items():
            folder_counts[folder.name][status] += 1
            if idx not in combined or priority[status] > priority[combined[idx]]:
                combined[idx] = status
                if idx in folder_timings:
                    combined_timings[idx] = folder_timings[idx]
            elif idx not in combined_timings and idx in folder_timings:
                combined_timings[idx] = folder_timings[idx]

    # Per-folder breakdown
    for fname, counts in folder_counts.items():
        total_in_folder = sum(counts.values())
        parts = ", ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
        print(f"  {fname}: {total_in_folder} files  ({parts})")
    print()

    # Overall counts
    status_counts: dict[str, int] = defaultdict(int)
    for status in combined.values():
        status_counts[status] += 1

    n_completed = status_counts["COMPLETED"]
    n_failed = status_counts["FAILED"]
    n_running = status_counts["RUNNING"]
    n_total_seen = len(combined)
    n_expected = args.total or n_total_seen

    n_not_seen = max(0, n_expected - n_total_seen)

    print("=" * 52)
    print(f"  COMPLETED : {n_completed:>6} / {n_expected}")
    print(f"  FAILED    : {n_failed:>6} / {n_expected}")
    print(f"  RUNNING   : {n_running:>6} / {n_expected}")
    if n_not_seen:
        print(f"  NOT SEEN  : {n_not_seen:>6} / {n_expected}  (no log file found)")
    print(f"  {'─'*36}")
    print(f"  Log files seen : {n_total_seen:>6} / {n_expected}")
    print("=" * 52)

    # Batch duration statistics (only over batches that finished COMPLETED
    # *and* for which a batch_total TIMING line was found)
    completed_durations = [
        combined_timings[idx]
        for idx, status in combined.items()
        if status == "COMPLETED" and idx in combined_timings
    ]

    if completed_durations:
        avg_s = sum(completed_durations) / len(completed_durations)
        min_s = min(completed_durations)
        max_s = max(completed_durations)
        n_missing_time = n_completed - len(completed_durations)

        print()
        print("Batch duration (COMPLETED batches, stage=batch_total):")
        print(f"  Average : {format_duration(avg_s)}  ({avg_s:.1f}s)")
        print(f"  Shortest: {format_duration(min_s)}  ({min_s:.1f}s)")
        print(f"  Longest : {format_duration(max_s)}  ({max_s:.1f}s)")
        print(f"  Based on: {len(completed_durations)} / {n_completed} completed batch(es)")
        if n_missing_time:
            print(f"  (no batch_total TIMING line found for {n_missing_time} completed batch(es))")
    else:
        print()
        print("Batch duration: no batch_total TIMING lines found among COMPLETED batches.")

    # Write output file
    out_path = Path(args.output)
    with open(out_path, "w") as fh:
        fh.write("# batch_index  status\n")
        for idx in sorted(combined.keys()):
            fh.write(f"{idx} {combined[idx]}\n")

    print(f"\nStatus file written: {out_path}  ({n_total_seen} entries)")


if __name__ == "__main__":
    main()