#!/usr/bin/env python3
"""
scan_job_status.py  —  Scan SLURM log folders for scatter, gather, or batch
                       job outcomes and write a status file for resume.

Supported stages
----------------
  scatter   One SLURM task processes N batch files serially.
            Index line  (head): "Task 0: processing 100 batch(es): 0 1 2 … 99"
            Status line (tail): WORKFLOW_TASK|…|phase=end|…|status=0
            Output:  one line per *batch* index (all N indices from the task).

  gather    One SLURM task processes one dump.
            Status line (tail): WORKFLOW_TASK|…|phase=end|…|status=0
            Output:  one line per dump index (task_id + offset).

  batch     Legacy mode (one task per batch).
            Index line  (head): BATCH_TASK|batch=N|phase=start
            Status line (tail): BATCH_STATUS|batch=N|status=ok
            Output:  one line per batch index.

Stage is auto-detected from the filename prefix inside each log folder
(scatter_*.out / gather_*.out / batch_*.out).  Use --stage to override.

Output format
-------------
  <index> COMPLETED|FAILED|RUNNING

This is the format consumed by build_pending_file() in submit_scatter_gather.sh.

Usage
-----
  # Scatter resume (auto-detect):
  python scan_job_status.py --log-root /scratch/logs  --output scatter_status.txt

  # Gather, explicit stage:
  python scan_job_status.py --log-root /scratch/logs  --stage gather \\
                             --output gather_status.txt

  # Specific folders:
  python scan_job_status.py --folders logs_2025-*/  --output scatter_status.txt

  # Show counts only (no output file):
  python scan_job_status.py --log-root /scratch/logs  --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAIL_LINES = 80
HEAD_LINES = 15

# ---------------------------------------------------------------------------
# Regex patterns — shared
# ---------------------------------------------------------------------------

# SLURM job report line present at the very end of every .out file.
# Matches COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY.
RE_JOB_REPORT = re.compile(
    r"Job Report:\s+\d+\s+\((?P<slurm_status>COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY)\)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Regex patterns — scatter / gather  (WORKFLOW_TASK lines)
# ---------------------------------------------------------------------------

# WORKFLOW_TASK|stage=scatter_batches|global_task=7|n_batches=100|phase=end|epoch=…|status=0
RE_WORKFLOW_END = re.compile(
    r"WORKFLOW_TASK\|[^|]*\|[^|]*\|[^|]*\|phase=end\|[^|]*\|status=(?P<status>\d+)"
)

# "Task 7: processing 100 batch(es): 0 1 2 … 99"
# Written by scatter_batches.slurm before the Python call.
RE_SCATTER_TASK_LINE = re.compile(
    r"^Task\s+\d+:\s+processing\s+\d+\s+batch\(es\):\s+(?P<indices>[\d\s]+)"
)

# WORKFLOW_TASK|stage=…|global_task=7|… or dump_index=7|…
RE_WORKFLOW_START_IDX = re.compile(
    r"WORKFLOW_TASK\|[^|]*\|(?:global_task|dump_index)=(?P<idx>\d+)\|"
)

# ---------------------------------------------------------------------------
# Regex patterns — batch  (legacy)
# ---------------------------------------------------------------------------

RE_BATCH_STATUS_OK = re.compile(r"BATCH_STATUS\|batch=\d+\|status=ok\b")
RE_BATCH_INDEX = re.compile(r"BATCH_TASK\|batch=(?P<idx>\d+)\|phase=start")

# ---------------------------------------------------------------------------
# Filename patterns
# ---------------------------------------------------------------------------

RE_OUT_FILENAME = re.compile(
    r"^(?P<prefix>scatter|gather|batch)_\d+_(?P<task_id>\d+)\.out$"
)

# Folder discovery: match logs_<anything> so both  logs_12345_foo  and
# logs_2025-05-11_14-00-00  are found.
RE_LOG_FOLDER = re.compile(r"^logs_")

# Metadata offset file written by submit scripts.
RE_META_OFFSET = re.compile(r"^#?\s*chunk_offset\s*=\s*(?P<val>\d+)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------

def _read_chunk(path: Path, from_end: bool, n_lines: int) -> list[str]:
    chunk_bytes = 4096 * n_lines
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        with open(path, "rb") as fh:
            if from_end:
                fh.seek(max(0, size - chunk_bytes))
            raw = fh.read(chunk_bytes)
    except OSError:
        return []
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return lines[-n_lines:] if from_end else lines[:n_lines]


def tail_lines(path: Path, n: int = TAIL_LINES) -> list[str]:
    return _read_chunk(path, from_end=True, n_lines=n)


def head_lines(path: Path, n: int = HEAD_LINES) -> list[str]:
    return _read_chunk(path, from_end=False, n_lines=n)


# ---------------------------------------------------------------------------
# Per-stage classifiers
# ---------------------------------------------------------------------------

def _slurm_status(tail: list[str]) -> str | None:
    """Return the SLURM terminal status string, or None if job is still running."""
    text = "\n".join(tail)
    m = RE_JOB_REPORT.search(text)
    return m.group("slurm_status").upper() if m else None


def classify_scatter(path: Path) -> tuple[str, list[int]]:
    """
    Returns (status, batch_indices).

    status        : 'COMPLETED' | 'FAILED' | 'RUNNING'
    batch_indices : list of individual batch indices owned by this task.
                    All are marked with the same status because the task
                    processes them as a unit — a partial failure reruns all
                    (file-existence-based resume handles partial completions).
    """
    head = head_lines(path)
    tail = tail_lines(path)
    tail_text = "\n".join(tail)

    # Extract the list of batch indices from the "Task N: processing …" line.
    indices: list[int] = []
    for line in head:
        m = RE_SCATTER_TASK_LINE.match(line.strip())
        if m:
            indices = [int(x) for x in m.group("indices").split()]
            break

    # Determine success: pipeline must have emitted status=0.
    workflow_end = RE_WORKFLOW_END.search(tail_text)
    pipeline_ok = workflow_end is not None and workflow_end.group("status") == "0"

    slurm_st = _slurm_status(tail)

    if slurm_st == "COMPLETED" and pipeline_ok:
        status = "COMPLETED"
    elif slurm_st is not None:
        status = "FAILED"
    else:
        status = "RUNNING"

    return status, indices


def classify_gather(path: Path, task_id: int, offset: int) -> tuple[str, list[int]]:
    """
    Returns (status, [dump_index]).

    Dump index is resolved from the WORKFLOW_TASK start line if present,
    otherwise falls back to offset + task_id.
    """
    head = head_lines(path)
    tail = tail_lines(path)
    tail_text = "\n".join(tail)

    # Try to read dump index from the log itself.
    dump_idx: int | None = None
    for line in head:
        m = RE_WORKFLOW_START_IDX.search(line)
        if m:
            dump_idx = int(m.group("idx"))
            break
    if dump_idx is None:
        dump_idx = offset + task_id

    workflow_end = RE_WORKFLOW_END.search(tail_text)
    pipeline_ok = workflow_end is not None and workflow_end.group("status") == "0"

    slurm_st = _slurm_status(tail)

    if slurm_st == "COMPLETED" and pipeline_ok:
        status = "COMPLETED"
    elif slurm_st is not None:
        status = "FAILED"
    else:
        status = "RUNNING"

    return status, [dump_idx]


def classify_batch(path: Path, task_id: int, offset: int) -> tuple[str, list[int]]:
    """Legacy batch classifier — preserves original scan_batch_status logic."""
    head = head_lines(path)
    tail = tail_lines(path)
    tail_text = "\n".join(tail)

    pipeline_ok = bool(RE_BATCH_STATUS_OK.search(tail_text))

    # Index from log, then fall back to offset + task_id.
    batch_idx: int | None = None
    for line in head:
        m = RE_BATCH_INDEX.search(line)
        if m:
            batch_idx = int(m.group("idx"))
            break
    if batch_idx is None:
        batch_idx = offset + task_id

    slurm_st = _slurm_status(tail)

    if slurm_st == "COMPLETED" and pipeline_ok:
        status = "COMPLETED"
    elif slurm_st is not None:
        status = "FAILED"
    else:
        status = "RUNNING"

    return status, [batch_idx]


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

def read_chunk_offset(folder: Path) -> int | None:
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


def detect_stage(folder: Path) -> str | None:
    """Infer stage from the filename prefix of .out files in the folder."""
    for path in folder.glob("*.out"):
        m = RE_OUT_FILENAME.match(path.name)
        if m:
            return m.group("prefix")   # 'scatter', 'gather', or 'batch'
    return None


def scan_folder(
    folder: Path,
    stage_override: str | None = None,
) -> tuple[str, dict[int, str]]:
    """
    Scan one log folder.

    Returns (stage_name, {index: status}).
    """
    stage = stage_override or detect_stage(folder)
    if stage is None:
        return ("unknown", {})

    offset = read_chunk_offset(folder) or 0
    results: dict[int, str] = {}
    priority = {"COMPLETED": 2, "FAILED": 1, "RUNNING": 0}

    for out_path in sorted(folder.glob(f"{stage}_*.out")):
        m = RE_OUT_FILENAME.match(out_path.name)
        if not m or m.group("prefix") != stage:
            continue
        task_id = int(m.group("task_id"))

        if stage == "scatter":
            status, indices = classify_scatter(out_path)
            if not indices:
                # Log line missing — fall back: mark the single task_id slot.
                # Resume-by-file-existence handles the rest safely.
                print(
                    f"  WARNING: could not parse batch indices from {out_path.name}; "
                    f"task_id={task_id} recorded as a placeholder.",
                    file=sys.stderr,
                )
                indices = [offset + task_id]
        elif stage == "gather":
            status, indices = classify_gather(out_path, task_id, offset)
        else:
            status, indices = classify_batch(out_path, task_id, offset)

        for idx in indices:
            if idx not in results or priority[status] > priority[results[idx]]:
                results[idx] = status

    return stage, results


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def discover_log_folders(log_root: Path) -> list[Path]:
    """Return all subdirectories matching logs_* under log_root."""
    return sorted(
        entry for entry in log_root.iterdir()
        if entry.is_dir() and RE_LOG_FOLDER.match(entry.name)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan SLURM log folders for scatter/gather/batch job outcomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "log_root_pos",
        metavar="LOG_ROOT",
        nargs="?",
        default=None,
        help="Root logs directory (positional shortcut for --log-root).",
    )
    parser.add_argument(
        "--log-root",
        metavar="DIR",
        default=None,
        help="Root logs directory.  All logs_* subdirs will be scanned.",
    )
    parser.add_argument(
        "--folders",
        metavar="DIR",
        nargs="+",
        default=None,
        help="Explicit list of log folder paths (alternative to --log-root).",
    )
    parser.add_argument(
        "--stage",
        choices=["scatter", "gather", "batch"],
        default=None,
        help="Force a specific stage instead of auto-detecting from filenames.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default="job_status.txt",
        help="Path to write the status file (default: job_status.txt).",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        metavar="N",
        help="Expected total number of indices (for summary denominator).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write the output file.",
    )
    args = parser.parse_args()

    log_root_str = args.log_root or args.log_root_pos
    if not log_root_str and not args.folders:
        parser.error("Provide LOG_ROOT (positional or --log-root) or use --folders.")

    if log_root_str:
        log_root = Path(log_root_str).resolve()
        if not log_root.is_dir():
            print(f"Error: log root not found: {log_root}", file=sys.stderr)
            sys.exit(1)
        folders = discover_log_folders(log_root)
        if not folders:
            print(f"No logs_* subdirectories found under {log_root}", file=sys.stderr)
            sys.exit(1)
    else:
        folders = [Path(f).resolve() for f in args.folders]

    print(f"Scanning {len(folders)} log folder(s):")
    for f in folders:
        print(f"  {f}")
    print()

    combined: dict[int, str] = {}
    priority = {"COMPLETED": 2, "FAILED": 1, "RUNNING": 0}
    detected_stages: set[str] = set()

    for folder in folders:
        stage, folder_results = scan_folder(folder, stage_override=args.stage)
        detected_stages.add(stage)

        counts: dict[str, int] = defaultdict(int)
        for idx, status in folder_results.items():
            counts[status] += 1
            if idx not in combined or priority[status] > priority[combined[idx]]:
                combined[idx] = status

        total_in_folder = sum(counts.values())
        parts = ", ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
        print(f"  {folder.name} [{stage}]: {total_in_folder} indices  ({parts})")

    if len(detected_stages) > 1:
        print(
            f"\n  WARNING: mixed stages detected across folders: {detected_stages}. "
            f"Use --stage to force one.",
            file=sys.stderr,
        )

    print()

    status_counts: dict[str, int] = defaultdict(int)
    for status in combined.values():
        status_counts[status] += 1

    n_completed = status_counts["COMPLETED"]
    n_failed    = status_counts["FAILED"]
    n_running   = status_counts["RUNNING"]
    n_seen      = len(combined)
    n_expected  = args.total or n_seen
    n_not_seen  = max(0, n_expected - n_seen)

    stage_label = "/".join(sorted(detected_stages)) or "unknown"
    print(f"{'=' * 52}")
    print(f"  Stage     : {stage_label}")
    print(f"  COMPLETED : {n_completed:>6} / {n_expected}")
    print(f"  FAILED    : {n_failed:>6} / {n_expected}")
    print(f"  RUNNING   : {n_running:>6} / {n_expected}")
    if n_not_seen:
        print(f"  NOT SEEN  : {n_not_seen:>6} / {n_expected}  (no log file found)")
    print(f"  {'─' * 36}")
    print(f"  Log entries seen : {n_seen:>6} / {n_expected}")
    print(f"{'=' * 52}")

    if args.dry_run:
        print("\n(dry-run: no output file written)")
        return

    out_path = Path(args.output)
    with open(out_path, "w") as fh:
        fh.write("# index  status\n")
        for idx in sorted(combined.keys()):
            fh.write(f"{idx} {combined[idx]}\n")

    print(f"\nStatus file written: {out_path}  ({n_seen} entries)")


if __name__ == "__main__":
    main()