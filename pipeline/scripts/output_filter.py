"""
Output filtering and logging for tracing runner.

Captures phantomanalysis stdout/stderr, filters known warnings, tracks progress,
and logs to file.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FilteredOutput:
    """Container for filtered output statistics and content."""

    suppressed_errors: dict[str, int] = field(default_factory=dict)
    suppressed_warnings: dict[str, int] = field(default_factory=dict)
    useful_lines: list[str] = field(default_factory=list)
    dump_count: int = 0


class OutputFilter:
    """Filter phantomanalysis output and suppress known warnings."""

    # Errors to suppress (exact match patterns)
    SUPPRESSED_ERROR_PATTERNS = [
        r"ERROR: could not find cs_min in header",
        r"ERROR: could not find mtot_in in header",
    ]

    # Warnings to suppress (contains patterns)
    SUPPRESSED_WARNING_PATTERNS = [
        r"WARNING! sink particle vwind not found",
        r"WARNING! sink particle Twind not found",
        r"WARNING! sink particle ieject not found",
        r"WARNING! sink particle sftype not found",
        r"WARNING! sink particle nseed not found",
        r"WARNING! sink particle Rbondi not found",
        r"WARNING! sink particle Pr_Bondi not found",
    ]

    # Verbose output patterns to suppress when verbose=False
    VERBOSE_OUTPUT_PATTERNS = [
        r"^\s*Input file name is",
        r"not first step data",
        r"Loaded A_V for",
        r"Analysis of.*complete",
        r"^>>>.*reading setup from file",
        r"FT:Phantom:",
        r"reading particles",
        r"got\s+\d+\s+sink properties",
        r"ID\|.*Mass.*\|",
        r"^\s*-+$",  # separator lines
        r"^\s*\d+\|",  # particle property table rows
        r"finished reading",
    ]

    def __init__(self, verbose: bool = False):
        """Initialize filter with verbosity setting."""
        self.verbose = verbose
        self.result = FilteredOutput()
        self.compiled_error_patterns = [re.compile(p, re.IGNORECASE) for p in self.SUPPRESSED_ERROR_PATTERNS]
        self.compiled_warning_patterns = [re.compile(p, re.IGNORECASE) for p in self.SUPPRESSED_WARNING_PATTERNS]
        self.compiled_verbose_patterns = [re.compile(p) for p in self.VERBOSE_OUTPUT_PATTERNS]

    def _is_suppressed_error(self, line: str) -> Optional[str]:
        """Check if line matches a suppressed error pattern. Returns matching pattern key."""
        for pattern in self.compiled_error_patterns:
            if pattern.search(line):
                # Extract the error type for tracking
                match = re.search(r"could not find (\w+)", line, re.IGNORECASE)
                if match:
                    return f"cs_min/mtot_in"
        return None

    def _is_suppressed_warning(self, line: str) -> Optional[str]:
        """Check if line matches a suppressed warning pattern. Returns warning key."""
        for pattern in self.compiled_warning_patterns:
            if pattern.search(line):
                # Extract warning type
                match = re.search(r"WARNING! sink particle (\w+)", line, re.IGNORECASE)
                if match:
                    return match.group(1)
        return None

    def _is_verbose_output(self, line: str) -> bool:
        """Check if line is verbose output that should be suppressed."""
        if self.verbose:
            return False  # Keep all output if verbose mode is on
        for pattern in self.compiled_verbose_patterns:
            if pattern.search(line):
                return True
        return False

    def _count_dump_processing(self, line: str) -> None:
        """Track dump processing from output."""
        if "Analysis of" in line and "complete" in line:
            self.result.dump_count += 1

    def filter_line(self, line: str) -> Optional[str]:
        """
        Filter a single line of output.

        Returns the line if it should be kept, None if it should be suppressed.
        Updates statistics for suppressed content.
        """
        line = line.rstrip("\n\r")
        if not line.strip():
            return None  # Skip empty lines

        # Check for suppressed errors
        error_key = self._is_suppressed_error(line)
        if error_key:
            self.result.suppressed_errors[error_key] = self.result.suppressed_errors.get(error_key, 0) + 1
            return None

        # Check for suppressed warnings
        warning_key = self._is_suppressed_warning(line)
        if warning_key:
            self.result.suppressed_warnings[warning_key] = self.result.suppressed_warnings.get(warning_key, 0) + 1
            return None

        # Check for verbose output
        if self._is_verbose_output(line):
            return None

        # Count dumps and keep useful output
        self._count_dump_processing(line)
        self.result.useful_lines.append(line)
        return line

    def filter_output(self, output: str) -> str:
        """Filter multi-line output and return kept lines."""
        filtered_lines = []
        for line in output.split("\n"):
            kept = self.filter_line(line)
            if kept:
                filtered_lines.append(kept)
        return "\n".join(filtered_lines)

    def get_suppression_summary(self) -> str:
        """Generate a summary of suppressed warnings and errors."""
        if not self.result.suppressed_errors and not self.result.suppressed_warnings:
            return ""

        lines = ["\n--- Suppressed Output Summary ---"]

        if self.result.suppressed_errors:
            error_items = ", ".join(sorted(self.result.suppressed_errors.keys()))
            lines.append(f"ERROR: could not find [{error_items}] in header")

        if self.result.suppressed_warnings:
            warning_items = ", ".join(sorted(self.result.suppressed_warnings.keys()))
            lines.append(f"WARNING! sink particle [{warning_items}] not found")

        lines.append("--- End Suppressed Summary ---\n")
        return "\n".join(lines)

    def write_log(self, log_file: Path, total_dumps: int) -> None:
        """Write filtered output and summary to log file."""
        with open(log_file, "w", encoding="utf-8") as f:
            # Write useful output
            for line in self.result.useful_lines:
                f.write(line + "\n")

            # Write suppression summary
            f.write(self.get_suppression_summary())

            # Write statistics
            f.write("\n--- Processing Statistics ---\n")
            f.write(f"Total dumps processed: {self.result.dump_count}/{total_dumps}\n")
            f.write(f"Suppressed error lines: {sum(self.result.suppressed_errors.values())}\n")
            f.write(f"Suppressed warning lines: {sum(self.result.suppressed_warnings.values())}\n")


class ProgressTracker:
    """Track and display progress of dump processing."""

    def __init__(self, total_dumps: int, report_every: int = 100):
        """Initialize with total number of dumps to process."""
        self.total_dumps = total_dumps
        self.current_dump = 0
        self.report_every = max(1, int(report_every))

    def update(self, line: str) -> bool:
        """Update progress if line indicates a completed dump.

        Returns True only when progress advanced.
        """
        if "Analysis of" in line and "complete" in line:
            self.current_dump += 1
            if self.current_dump % self.report_every == 0 or self.current_dump == self.total_dumps:
                return True
        return False

    def get_progress_string(self) -> str:
        """Return a compact progress representation."""
        percent = (self.current_dump / self.total_dumps * 100) if self.total_dumps > 0 else 0
        bar_length = 30
        filled = int(bar_length * self.current_dump / self.total_dumps) if self.total_dumps > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        return f"[{bar}] {self.current_dump}/{self.total_dumps} ({percent:.0f}%)"

    def print_progress(self, file=sys.stderr) -> None:
        """Render progress in-place on a single line."""
        print(f"\rProgress: {self.get_progress_string()}", end="", file=file, flush=True)

    def finish(self, file=sys.stderr) -> None:
        """Terminate in-place progress line with a newline."""
        if self.current_dump > 0:
            print("", file=file, flush=True)
