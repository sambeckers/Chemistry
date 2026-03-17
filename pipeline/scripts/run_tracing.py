from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from common import BatchPlanner, DumpSelection, FileSystemTools, ParticleIdStore, PipelineConfigManager
from output_filter import OutputFilter, ProgressTracker


class TracingRunner:
    """Run one global phantomanalysis call and stage per-batch binary trace files."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = PipelineConfigManager.load(self.config_path)

        paths = self.config["paths"]
        self.processing = self.config["processing"]
        self.prefix = self.config["simulation"]["prefix"]
        self.verbose = bool(self.processing.get("verbose_tracing", False))
        self.auto_compile_analysis = bool(self.processing.get("auto_compile_analysis", True))
        self.force_recompile_analysis = bool(self.processing.get("force_recompile_analysis", False))
        self.analysis_compile_system = str(self.processing.get("analysis_compile_system", "gfortran"))

        self.dump_dir = Path(paths["phantom_dump_dir"]).resolve()
        self.av_dir = Path(paths["av_dir"]).resolve()
        self.phantomanalysis_binary = Path(paths["phantomanalysis_binary"]).resolve()
        self.analysis_trace_source = Path(
            paths.get("analysis_trace_source", "/fred/oz304/beckers/phantom/src/utils/analysis_trace.f90")
        ).resolve()
        self.phantom_writemake_script = Path(
            paths.get("phantom_writemake_script", "/fred/oz304/beckers/phantom/scripts/writemake.sh")
        ).resolve()
        self.scratch_root = Path(paths["scratch_root"]).resolve()

        default_root = self.scratch_root / "tracing"
        self.tracing_root = Path(paths.get("tracing_root", default_root)).resolve()
        self.trace_batches_dir = self.tracing_root / "batches"
        self.trace_binary_dir = self.tracing_root / "batches"
        self.trace_work_dir = self.tracing_root / "work"
        self.trace_metadata_file = self.tracing_root / "tracing_metadata.yaml"
        
        # Keep filtered tracing log with tracing artifacts, not in a separate logs/tracing_* folder.
        self.trace_log_file = self.tracing_root / f"tracing_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def _compile_required(self) -> tuple[bool, str]:
        """Determine whether phantomanalysis should be rebuilt before tracing."""
        if self.force_recompile_analysis:
            return True, "forced by processing.force_recompile_analysis"
        if not self.phantomanalysis_binary.is_file():
            return True, "phantomanalysis binary missing"
        source_mtime = self.analysis_trace_source.stat().st_mtime
        binary_mtime = self.phantomanalysis_binary.stat().st_mtime
        if source_mtime > binary_mtime:
            return True, "analysis_trace source is newer than phantomanalysis binary"
        return False, "binary is up to date"

    def _compile_phantomanalysis(self, reason: str) -> None:
        """Rebuild phantomanalysis analysis target in the *_out directory."""
        if not self.phantom_writemake_script.is_file():
            raise FileNotFoundError(f"writemake.sh not found: {self.phantom_writemake_script}")

        build_dir = self.phantomanalysis_binary.parent
        if not build_dir.is_dir():
            raise FileNotFoundError(f"phantomanalysis build directory not found: {build_dir}")

        print(f"Compiling phantomanalysis in {build_dir} ({reason})")
        makefile_path = build_dir / "Makefile"
        with open(makefile_path, "w", encoding="ascii") as makefile_handle:
            subprocess.run(
                [str(self.phantom_writemake_script), self.prefix],
                cwd=build_dir,
                stdout=makefile_handle,
                check=True,
            )

        subprocess.run(
            [
                "make",
                "analysis",
                f"ANALYSIS={self.analysis_trace_source.name}",
                f"SYSTEM={self.analysis_compile_system}",
            ],
            cwd=build_dir,
            check=True,
        )

        if not self.phantomanalysis_binary.is_file():
            raise RuntimeError("Compilation completed but phantomanalysis binary was not produced")
        self.phantomanalysis_binary.chmod(0o755)
        print(f"Compilation complete: {self.phantomanalysis_binary}")

    def _verify_analysis_trace_compatibility(self) -> None:
        """Sanity-check analysis_trace source for expected trace binary record layout."""
        source_text = self.analysis_trace_source.read_text(encoding="ascii", errors="replace")
        normalized = re.sub(r"\s+", " ", source_text.lower())

        expected_write = (
            "write(batch_units(batch_idx+1)) pid_out, time_out, x_out, y_out, z_out, "
            "density_out, temp_out, av_out"
        )
        if expected_write.lower() not in normalized:
            raise RuntimeError(
                "analysis_trace.f90 no longer writes the expected record layout "
                "(pid,time,x,y,z,density,temp,av) required by pipeline binary reader"
            )

        required_tokens = ["trace_config", "batch_map_file", "n_batches", "id_start", "id_end"]
        missing = [token for token in required_tokens if token.lower() not in normalized]
        if missing:
            raise RuntimeError(
                "analysis_trace.f90 appears incompatible with pipeline trace.cfg/batch-map flow. "
                f"Missing tokens: {', '.join(missing)}"
            )

    @staticmethod
    def _ensure_link(source: Path, destination: Path) -> None:
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.symlink(source, destination)

    @staticmethod
    def _write_trace_cfg(
        path: Path,
        id_start: int,
        id_end: int,
        batch_count: int,
        batch_map_file: str,
    ) -> None:
        path.write_text(
            "&trace_config\n"
            f"  id_start = {id_start}\n"
            f"  id_end = {id_end}\n"
            f"  n_batches = {batch_count}\n"
            f"  batch_map_file = '{batch_map_file}'\n"
            "/\n",
            encoding="ascii",
        )

    @staticmethod
    def _write_batch_map(path: Path, selected_ids: np.ndarray, layout: list[tuple[int, int]]) -> None:
        """Write particle->batch map consumed by analysis_trace.f90."""
        with open(path, "w", encoding="ascii") as handle:
            for batch_index, (start, end) in enumerate(layout):
                for pid in selected_ids[start:end].tolist():
                    handle.write(f"{int(pid)} {int(batch_index)}\n")

    def _load_particles_and_layout(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        ids_all = ParticleIdStore.load_cached(self.config)
        if ids_all is None:
            raise RuntimeError(
                "Particle ID cache is missing/stale. Run ./workflow/submit_discover_particle_ids.sh first."
            )
        n_boundary = int(self.processing.get("n_boundary", 0))
        selected_ids = ParticleIdStore.apply_n_boundary(ids_all, n_boundary)
        layout = BatchPlanner.compute_layout(
            total_particles=int(len(selected_ids)),
            batch_size=self.processing.get("batch_size"),
            n_batches=self.processing.get("n_batches"),
        )
        if not layout:
            raise RuntimeError("Batch layout is empty; check processing.batch_size/n_batches")
        return selected_ids, layout

    @staticmethod
    def _planned_ids_for_layout(selected_ids: np.ndarray, layout: list[tuple[int, int]]) -> np.ndarray:
        """Return only IDs that are actually covered by the configured batch layout."""
        max_end = int(layout[-1][1])
        if max_end <= 0:
            raise RuntimeError("Invalid batch layout: max_end must be > 0")
        return selected_ids[:max_end]

    def _run_phantomanalysis_once(self, selected_ids: np.ndarray, selected_dumps: list[int], layout: list[tuple[int, int]]) -> Path:
        if selected_ids.size == 0:
            raise RuntimeError("No selected particle IDs for tracing")

        FileSystemTools.ensure_clean_directory(self.tracing_root)
        if self.trace_work_dir.exists():
            shutil.rmtree(self.trace_work_dir)
        self.trace_work_dir.mkdir(parents=True, exist_ok=True)

        trace_output_dir = self.trace_work_dir / "trace_output"
        trace_output_dir.mkdir(parents=True, exist_ok=True)

        batch_map_path = self.trace_work_dir / "trace_batch_map.txt"
        self._write_batch_map(batch_map_path, selected_ids=selected_ids, layout=layout)

        self._write_trace_cfg(
            self.trace_work_dir / "trace.cfg",
            id_start=int(selected_ids[0]),
            id_end=int(selected_ids[-1]),
            batch_count=len(layout),
            batch_map_file=batch_map_path.name,
        )
        self._ensure_link(self.av_dir, self.trace_work_dir / "AV")

        binary = self.trace_work_dir / "phantomanalysis"
        shutil.copy2(self.phantomanalysis_binary, binary)
        binary.chmod(0o755)

        dump_paths = [str(self.dump_dir / f"{self.prefix}_{dump:05d}") for dump in selected_dumps]
        
        # Run phantomanalysis with output capture and filtering
        print(
            f"\nParticle range: IDs {int(selected_ids[0])} to {int(selected_ids[-1])} ({len(selected_ids)} total)",
            flush=True,
        )
        print(f"Processing {len(selected_dumps)} dumps...", flush=True)
        
        output_filter = OutputFilter(verbose=self.verbose)
        progress_tracker = ProgressTracker(len(selected_dumps), report_every=100)
        
        try:
            process = subprocess.Popen(
                [str(binary), *dump_paths],
                cwd=self.trace_work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            for line in process.stdout:
                # Filter and display output
                filtered = output_filter.filter_line(line)
                progressed = progress_tracker.update(line)
                
                # Show progress only when a dump completed.
                if progressed:
                    progress_tracker.print_progress()
                
                # Print useful output (if verbose or if not suppressed)
                if filtered and self.verbose:
                    print(f"\n{filtered}", file=sys.stderr)
            
            return_code = process.wait()
            
            if return_code != 0:
                raise RuntimeError(f"phantomanalysis failed with exit code {return_code}")
        finally:
            progress_tracker.finish()

            # Write log file with summary
            output_filter.write_log(self.trace_log_file, len(selected_dumps))
            
            # Print suppression summary to console
            summary = output_filter.get_suppression_summary()
            if summary:
                print(summary)
            
            print(f"Tracing log saved to: {self.trace_log_file}")
        
        return trace_output_dir

    @staticmethod
    def _read_batch_binary_records(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Missing batch trace binary: {path}")
        record_dtype = np.dtype(
            [
                ("pid", np.int64),
                ("time", np.float64),
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("density", np.float64),
                ("temp", np.float64),
                ("av", np.float64),
            ]
        )
        return np.fromfile(path, dtype=record_dtype)

    @staticmethod
    def _format_phys_line(record: np.void) -> str:
        return (
            f"{float(record['time']):16.8E} "
            f"{float(record['x']):14.7E} "
            f"{float(record['y']):14.7E} "
            f"{float(record['z']):14.7E} "
            f"{float(record['density']):14.7E} "
            f"{float(record['temp']):8.2f} "
            f"{float(record['av']):10.8f}\n"
        )

    def _stage_batch_binaries(self, trace_output_dir: Path, batch_count: int) -> Path:
        """Move per-batch binaries from temporary trace output into persistent tracing storage."""
        if self.trace_binary_dir.exists():
            shutil.rmtree(self.trace_binary_dir)
        self.trace_binary_dir.mkdir(parents=True, exist_ok=True)

        for batch_index in range(batch_count):
            src = trace_output_dir / f"trace_batch_{batch_index:05d}.bin"
            if not src.is_file():
                raise FileNotFoundError(f"Missing batch trace binary: {src}")
            dst = self.trace_binary_dir / src.name
            shutil.move(str(src), str(dst))
        return self.trace_binary_dir



    def _write_metadata_and_update_config(
        self,
        selected_ids: np.ndarray,
        layout: list[tuple[int, int]],
        selected_dumps: list[int],
    ) -> None:
        payload = {
            "trace_binary_batches_dir": str(self.trace_binary_dir.resolve()),
            "batch_count": int(len(layout)),
            "particle_count": int(len(selected_ids)),
            "dump_count": int(len(selected_dumps)),
            "dump_numbers": [int(d) for d in selected_dumps],
        }
        self.trace_metadata_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="ascii")

        with open(self.config_path, "r", encoding="ascii") as handle:
            cfg = yaml.safe_load(handle)
        if "paths" not in cfg:
            cfg["paths"] = {}
        cfg["paths"]["trace_binary_batches_dir"] = str(self.trace_binary_dir.resolve())
        cfg["paths"]["trace_metadata_file"] = str(self.trace_metadata_file.resolve())
        cfg["paths"]["analysis_trace_source"] = str(self.analysis_trace_source)
        with open(self.config_path, "w", encoding="ascii") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

    def run(self) -> None:
        if not self.analysis_trace_source.is_file():
            raise FileNotFoundError(f"analysis_trace source file not found: {self.analysis_trace_source}")

        self._verify_analysis_trace_compatibility()
        if self.auto_compile_analysis:
            should_compile, reason = self._compile_required()
            if should_compile:
                self._compile_phantomanalysis(reason)
            else:
                print(f"Skipping compile: {reason}")
        else:
            print("Auto-compile disabled (processing.auto_compile_analysis=false); using existing phantomanalysis binary")

        selected_dumps = DumpSelection.selected_dump_numbers(self.config)
        selected_ids, layout = self._load_particles_and_layout()
        planned_ids = self._planned_ids_for_layout(selected_ids, layout)

        print(
            f"Tracing only planned batch IDs: {len(planned_ids)} selected "
            f"(from {len(selected_ids)} discovered after n_boundary)"
        )
        print("Tracing boundary mode: boundary filtering done in Python selection; Fortran uses particle type only")

        print(f"Tracing with analysis source: {self.analysis_trace_source}")
        trace_output_dir = self._run_phantomanalysis_once(
            selected_ids=planned_ids,
            selected_dumps=selected_dumps,
            layout=layout,
        )
        trace_binary_dir = self._stage_batch_binaries(trace_output_dir=trace_output_dir, batch_count=len(layout))
        self._write_metadata_and_update_config(
            selected_ids=planned_ids,
            layout=layout,
            selected_dumps=selected_dumps,
        )

        if bool(self.processing.get("cleanup_temporary", True)) and self.trace_work_dir.exists():
            shutil.rmtree(self.trace_work_dir)

        print(json.dumps({
            "status": "ok",
            "trace_binary_batches_dir": str(trace_binary_dir.resolve()),
            "trace_metadata_file": str(self.trace_metadata_file.resolve()),
            "batch_count": int(len(layout)),
        }))


class TracingCLI:
    """CLI front-end for one-shot tracing."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Run one global tracing call and stage per-batch binary trace files.")
        parser.add_argument("--config", required=True)
        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        args = cls.parse_args()
        TracingRunner(config_path=Path(args.config)).run()


def main() -> None:
    TracingCLI.run()


if __name__ == "__main__":
    main()
