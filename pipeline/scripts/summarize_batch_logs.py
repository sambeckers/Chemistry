from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

TIMING_RE = re.compile(r"^TIMING\|batch=(\d+)\|stage=([^|]+)\|seconds=([0-9]*\.?[0-9]+)\s*$")
STATUS_RE = re.compile(
    r"^BATCH_STATUS\|batch=(\d+)\|status=([a-z]+)\|seconds=([0-9]*\.?[0-9]+)(?:\|error=([^|\n]+))?\s*$"
)
TASK_RE = re.compile(r"^BATCH_TASK\|batch=(\d+)\|phase=(start|end)\|epoch=(\d+)(?:\|status=(\d+))?\s*$")
WORKFLOW_TASK_RE = re.compile(
    r"^WORKFLOW_TASK\|stage=([^|]+)(?:\|item=([^|]+))?\|phase=(start|end)\|epoch=(\d+)(?:\|status=(\d+))?\s*$"
)
ISSUE_RE = re.compile(r"\b(warning|warn|error|traceback|exception|failed|failure)\b", re.IGNORECASE)
WARNING_RE = re.compile(r"\b(warning|warn)\b", re.IGNORECASE)
ERROR_RE = re.compile(r"\b(error|traceback|exception|failed|failure)\b", re.IGNORECASE)
PARTICLE_RE = re.compile(r"^Batch\s+(\d+):\s+particles\s+\d+:\d+\s+\((\d+)\s+particle\(s\)\s+out\s+of\s+\d+\)")

STAGE_ALIASES = {
    "trace_stage": "tracing",
}

SUBPROCESS_STAGE_ORDER = [
    "prepare_batch_paths",
    "prepare_trace_workspace",
    "init_batch_hdf5",
    "prepare_chem_runtime",
    "phys_to_txt",
    "ev_convert",
    "append_hdf5",
    "cleanup_particle_products",
    "cleanup_batch_workspace",
]

MAIN_PROCESS_STAGE_ORDER = [
    "tracing",
    "chemistry_model",
    "batch_total",
]

PROCESS_TOTAL_STAGE_ORDER = [
    "tracing",
    "chemistry_model",
    *SUBPROCESS_STAGE_ORDER,
    "batch_total",
]


def _to_int_suffix(path: Path) -> int | None:
    stem = path.stem
    part = stem.rsplit("_", 1)
    if len(part) != 2:
        return None
    try:
        return int(part[1])
    except ValueError:
        return None


def _seconds_to_hms(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _seconds_to_hms_ms(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    hours, rem_ms = divmod(total_ms, 3600 * 1000)
    minutes, rem_ms = divmod(rem_ms, 60 * 1000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _canonical_stage_name(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _seconds_per_particle_str(seconds: float, particles: int) -> str:
    if particles <= 0:
        return "n/a"
    return _seconds_to_hms(seconds / float(particles))


def _seconds_per_particle_str_ms(seconds: float, particles: int) -> str:
    if particles <= 0:
        return "n/a"
    return _seconds_to_hms_ms(seconds / float(particles))


def _stage_seconds(stages: dict[str, float], stage_name: str) -> float:
    return stages.get(stage_name, 0.0)


def _workflow_stage_info() -> dict:
    return {
        "starts": [],
        "ends": [],
        "ok": 0,
        "failed": 0,
        "running": 0,
    }


def _render_aligned_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render an aligned plain-text table for easier scanning in logs."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def _format(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "  ".join("-" * width for width in widths)
    return [_format(headers), separator, *(_format(row) for row in rows)]


def main() -> None:
    log_dir = Path(os.environ.get("BATCH_LOG_DIR", "")).expanduser().resolve()
    array_job_id = os.environ.get("BATCH_ARRAY_JOB_ID", "unknown")
    summary_output_env = os.environ.get("SUMMARY_OUTPUT", "").strip()

    if not str(log_dir):
        raise RuntimeError("BATCH_LOG_DIR is required")
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Batch log directory not found: {log_dir}")

    summary_output = (
        Path(summary_output_env).expanduser().resolve()
        if summary_output_env
        else (log_dir / "batch_run_summary.txt").resolve()
    )

    out_logs = sorted(
        [*log_dir.glob("batch_*_*.out"), *log_dir.glob("chem_hdf5_batch_*_*.out")],
        key=lambda p: (p.name, p.stat().st_mtime),
    )
    err_logs = sorted(
        [*log_dir.glob("batch_*_*.err"), *log_dir.glob("chem_hdf5_batch_*_*.err")],
        key=lambda p: (p.name, p.stat().st_mtime),
    )

    batch_data: dict[int, dict] = {}
    workflow_tasks: dict[str, dict] = {
        "particle_discovery_worker": _workflow_stage_info(),
        "particle_discovery_merge": _workflow_stage_info(),
        "run_batch": _workflow_stage_info(),
        "merge_dumps": _workflow_stage_info(),
    }

    def get_batch(index: int) -> dict:
        info = batch_data.setdefault(
            index,
            {
                "stages": {},
                "warnings": 0,
                "errors": 0,
                "issue_lines": [],
                "status": "unknown",
                "status_seconds": None,
                "status_error": None,
                "task_start": None,
                "task_end": None,
                "task_exit_status": None,
                "particles": None,
            },
        )
        return info

    for out_path in out_logs:
        index = _to_int_suffix(out_path)
        if index is None:
            continue
        info = get_batch(index)
        for line in _read_lines(out_path):
            m = PARTICLE_RE.match(line)
            if m:
                info["particles"] = int(m.group(2))
                continue

            m = TIMING_RE.match(line)
            if m:
                stage = _canonical_stage_name(m.group(2))
                seconds = float(m.group(3))
                info["stages"][stage] = info["stages"].get(stage, 0.0) + seconds
                continue

            m = STATUS_RE.match(line)
            if m:
                info["status"] = m.group(2)
                info["status_seconds"] = float(m.group(3))
                info["status_error"] = m.group(4)
                continue

            m = TASK_RE.match(line)
            if m:
                phase = m.group(2)
                epoch = int(m.group(3))
                if phase == "start":
                    info["task_start"] = epoch
                    workflow_tasks["run_batch"]["starts"].append(epoch)
                else:
                    info["task_end"] = epoch
                    workflow_tasks["run_batch"]["ends"].append(epoch)
                    if m.group(4) is not None:
                        info["task_exit_status"] = int(m.group(4))
                        if info["task_exit_status"] == 0:
                            workflow_tasks["run_batch"]["ok"] += 1
                        else:
                            workflow_tasks["run_batch"]["failed"] += 1
                continue

            m = WORKFLOW_TASK_RE.match(line)
            if m:
                stage = m.group(1)
                phase = m.group(3)
                epoch = int(m.group(4))
                status_group = m.group(5)
                stage_info = workflow_tasks.setdefault(stage, _workflow_stage_info())
                if phase == "start":
                    stage_info["starts"].append(epoch)
                else:
                    stage_info["ends"].append(epoch)
                    if status_group is not None:
                        status = int(status_group)
                        if status == 0:
                            stage_info["ok"] += 1
                        else:
                            stage_info["failed"] += 1
                continue

            if ISSUE_RE.search(line):
                if WARNING_RE.search(line):
                    info["warnings"] += 1
                if ERROR_RE.search(line):
                    info["errors"] += 1
                if len(info["issue_lines"]) < 8:
                    info["issue_lines"].append(f"OUT: {line.strip()}")

    err_by_index = {_to_int_suffix(path): path for path in err_logs}
    for index, err_path in err_by_index.items():
        if index is None:
            continue
        info = get_batch(index)
        for line in _read_lines(err_path):
            if ISSUE_RE.search(line):
                if WARNING_RE.search(line):
                    info["warnings"] += 1
                if ERROR_RE.search(line):
                    info["errors"] += 1
                if len(info["issue_lines"]) < 8:
                    info["issue_lines"].append(f"ERR: {line.strip()}")

    workflow_out_logs = sorted(log_dir.glob("*.out"), key=lambda p: (p.name, p.stat().st_mtime))
    for out_path in workflow_out_logs:
        if out_path.name.startswith("batch_") or out_path.name.startswith("chem_hdf5_batch_"):
            continue
        for line in _read_lines(out_path):
            m = WORKFLOW_TASK_RE.match(line)
            if not m:
                continue
            stage = m.group(1)
            phase = m.group(3)
            epoch = int(m.group(4))
            status_group = m.group(5)
            stage_info = workflow_tasks.setdefault(stage, _workflow_stage_info())
            if phase == "start":
                stage_info["starts"].append(epoch)
            else:
                stage_info["ends"].append(epoch)
                if status_group is not None:
                    status = int(status_group)
                    if status == 0:
                        stage_info["ok"] += 1
                    else:
                        stage_info["failed"] += 1

    for stage_info in workflow_tasks.values():
        starts = len(stage_info["starts"])
        ends = len(stage_info["ends"])
        if starts > ends:
            stage_info["running"] = starts - ends

    if not batch_data:
        now = datetime.now(timezone.utc).isoformat()
        summary = [
            f"Batch Run Summary ({now})",
            f"Array job id: {array_job_id}",
            f"Log directory: {log_dir}",
            "No batch logs found.",
        ]
        summary_output.write_text("\n".join(summary) + "\n", encoding="ascii")
        print(f"Wrote summary: {summary_output}")
        return

    indices = sorted(batch_data.keys())
    all_starts = [batch_data[i]["task_start"] for i in indices if batch_data[i]["task_start"] is not None]
    all_ends = [batch_data[i]["task_end"] for i in indices if batch_data[i]["task_end"] is not None]

    total_wall_seconds = 0.0
    if all_starts and all_ends:
        total_wall_seconds = float(max(all_ends) - min(all_starts))

    total_status_seconds = sum(batch_data[i]["status_seconds"] or 0.0 for i in indices)
    total_particles = sum(int(batch_data[i]["particles"] or 0) for i in indices)
    total_warnings = sum(int(batch_data[i]["warnings"]) for i in indices)
    total_errors = sum(int(batch_data[i]["errors"]) for i in indices)

    summary_lines = []
    now = datetime.now(timezone.utc).isoformat()
    summary_lines.append(f"Batch Run Summary ({now})")
    summary_lines.append(f"Array job id: {array_job_id}")
    summary_lines.append(f"Log directory: {log_dir}")
    summary_lines.append("")
    summary_lines.append("Totals")
    summary_lines.append(f"- Batches discovered: {len(indices)}")
    summary_lines.append(f"- Total particles reported by batches: {total_particles}")
    summary_lines.append(
        "- Batch execution span on allocated workers "
        f"(first BATCH_TASK start -> last BATCH_TASK end, excludes scheduler pending): {_seconds_to_hms(total_wall_seconds)}"
    )
    summary_lines.append(
        "- Sum of per-batch runtimes from BATCH_STATUS "
        f"(adds batches together, so parallel overlap can make this larger than execution span): {_seconds_to_hms(total_status_seconds)}"
    )
    summary_lines.append(f"- Total warning lines: {total_warnings}")
    summary_lines.append(f"- Total error lines: {total_errors}")
    summary_lines.append("- Note: status/timing fields are unknown or 0.00 for legacy logs without TIMING/BATCH_STATUS markers")
    summary_lines.append("")

    summary_lines.append("Per-batch overview")
    summary_header = [
        "batch",
        "status",
        "particles",
        "batch_runtime_hms",
        "runtime_per_particle_hms",
        "execution_span_hms",
        "warnings",
        "errors",
        "exit_status",
    ]
    summary_rows: list[list[str]] = []

    for index in indices:
        info = batch_data[index]
        runtime_seconds = info["status_seconds"] or 0.0
        particles = int(info["particles"] or 0)
        if info["task_start"] is not None and info["task_end"] is not None:
            execution_span_hms = _seconds_to_hms(float(info["task_end"] - info["task_start"]))
        else:
            execution_span_hms = "n/a"
        exit_status = str(info["task_exit_status"]) if info["task_exit_status"] is not None else "n/a"
        summary_rows.append(
            [
                str(index),
                str(info["status"]),
                str(particles),
                _seconds_to_hms(runtime_seconds),
                _seconds_per_particle_str(runtime_seconds, particles),
                execution_span_hms,
                str(info["warnings"]),
                str(info["errors"]),
                exit_status,
            ]
        )
    summary_lines.extend(_render_aligned_table(summary_header, summary_rows))

    summary_lines.append("")
    summary_lines.append("Per-batch main-process timings")
    main_stage_header = [
        "batch",
        "tracing_hms",
        "chemistry_model_hms",
        "batch_total_hms",
    ]
    main_stage_rows: list[list[str]] = []
    for index in indices:
        info = batch_data[index]
        stages = info["stages"]
        main_stage_rows.append(
            [
                str(index),
                _seconds_to_hms(_stage_seconds(stages, 'tracing')),
                _seconds_to_hms(_stage_seconds(stages, 'chemistry_model')),
                _seconds_to_hms(_stage_seconds(stages, 'batch_total')),
            ]
        )
    summary_lines.extend(_render_aligned_table(main_stage_header, main_stage_rows))

    summary_lines.append("")
    summary_lines.append("Per-batch subprocess timings")
    subprocess_header = [
        "batch",
        "prepare_paths_hms_ms",
        "prepare_trace_ws_hms_ms",
        "init_batch_hdf5_hms_ms",
        "prepare_chem_runtime_hms_ms",
        "phys_to_txt_hms_ms",
        "ev_convert_hms_ms",
        "append_hdf5_hms_ms",
        "cleanup_particle_products_hms_ms",
        "cleanup_batch_workspace_hms_ms",
    ]
    subprocess_rows: list[list[str]] = []
    for index in indices:
        info = batch_data[index]
        stages = info["stages"]
        subprocess_rows.append(
            [
                str(index),
                _seconds_to_hms_ms(_stage_seconds(stages, 'prepare_batch_paths')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'prepare_trace_workspace')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'init_batch_hdf5')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'prepare_chem_runtime')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'phys_to_txt')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'ev_convert')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'append_hdf5')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'cleanup_particle_products')),
                _seconds_to_hms_ms(_stage_seconds(stages, 'cleanup_batch_workspace')),
            ]
        )
    summary_lines.extend(_render_aligned_table(subprocess_header, subprocess_rows))

    summary_lines.append("")
    summary_lines.append("Process runtime totals")
    process_header = ["process", "category", "total_runtime_hms", "runtime_per_particle_hms"]
    process_rows: list[list[str]] = []
    for stage_name in PROCESS_TOTAL_STAGE_ORDER:
        total_seconds = sum(_stage_seconds(batch_data[index]["stages"], stage_name) for index in indices)
        category = "main" if stage_name in MAIN_PROCESS_STAGE_ORDER else "subprocess"
        total_runtime = _seconds_to_hms(total_seconds) if category == "main" else _seconds_to_hms_ms(total_seconds)
        per_particle_runtime = (
            _seconds_per_particle_str(total_seconds, total_particles)
            if category == "main"
            else _seconds_per_particle_str_ms(total_seconds, total_particles)
        )
        process_rows.append(
            [
                stage_name,
                category,
                total_runtime,
                per_particle_runtime,
            ]
        )
    summary_lines.extend(_render_aligned_table(process_header, process_rows))

    summary_lines.append("")
    summary_lines.append("Workflow stage jobs")
    workflow_header = ["stage", "status", "tasks", "execution_span_hms", "notes"]
    workflow_rows: list[list[str]] = []

    metadata_path = log_dir / "workflow_submit_metadata.txt"
    metadata: dict[str, str] = {}
    if metadata_path.exists():
        for line in _read_lines(metadata_path):
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    def render_workflow_row(stage_key: str, label: str, fallback_note: str = "") -> None:
        info = workflow_tasks.get(stage_key, _workflow_stage_info())
        starts = info["starts"]
        ends = info["ends"]
        ok = int(info["ok"])
        failed = int(info["failed"])
        running = int(info["running"])
        task_count = max(len(starts), len(ends), ok + failed + running)

        if starts and ends:
            span_hms = _seconds_to_hms(float(max(ends) - min(starts)))
        else:
            span_hms = "n/a"

        if task_count == 0:
            status = "skipped"
            notes = fallback_note or "no task markers found"
        elif failed > 0:
            status = "failed"
            notes = f"ok={ok}, failed={failed}, running={running}"
        elif running > 0:
            status = "running"
            notes = f"ok={ok}, failed={failed}, running={running}"
        else:
            status = "ok"
            notes = f"ok={ok}, failed={failed}"

        workflow_rows.append([label, status, str(task_count), span_hms, notes])

    discover_note = ""
    if metadata.get("discovery_needed") == "0":
        discover_note = "skipped by cache-valid check in batch.sh"

    render_workflow_row("particle_discovery_worker", "particle_discovery_workers", discover_note)
    render_workflow_row("particle_discovery_merge", "particle_discovery_merge", discover_note)
    render_workflow_row("run_batch", "run_batch_array")
    render_workflow_row("merge_dumps", "merge_dumps")
    summary_lines.extend(_render_aligned_table(workflow_header, workflow_rows))

    summary_lines.append("")
    summary_lines.append("Warnings and Errors (sampled)")
    found_issue = False
    for index in indices:
        issues = batch_data[index]["issue_lines"]
        if not issues:
            continue
        found_issue = True
        summary_lines.append(f"- Batch {index}:")
        for issue_line in issues:
            summary_lines.append(f"  {issue_line}")
    if not found_issue:
        summary_lines.append("- None detected in batch stdout/stderr logs.")

    summary_output.write_text("\n".join(summary_lines) + "\n", encoding="ascii")
    print(f"Wrote summary: {summary_output}")


if __name__ == "__main__":
    main()
