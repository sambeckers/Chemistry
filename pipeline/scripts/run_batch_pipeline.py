from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

from append_ev_to_hdf5 import BatchHDF5Appender
from common import BatchPlanner, DumpSelection, FileSystemTools, SpeciesCatalog, PipelineConfigManager
from convert_phys_to_txt import PhysTraceConverter


FORTRAN_PATH_LIMIT = 70


class BatchPipelineRunner:
    """Run one chemistry pipeline batch from trace extraction to HDF5 append."""

    def __init__(self, config: dict, batch_index: int) -> None:
        """Store config and resolve batch particle selection metadata."""
        self.config = config
        self.batch_index = batch_index
        self.particle_ids, self.metadata = BatchPlanner.batch_particle_ids(config, batch_index)
        if len(self.particle_ids) == 0:
            raise RuntimeError(f"Batch {batch_index} contains no particles")

        self.processing = config["processing"]
        self.dump_numbers = DumpSelection.target_dump_numbers(config)
        self.chemistry_types = list(self.processing.get("chemistry_types", ["Crich"]))

        self.batch_output_dir = FileSystemTools.ensure_clean_directory(config["paths"]["batch_output_dir"])
        self.scratch_root = FileSystemTools.ensure_clean_directory(config["paths"]["scratch_root"])
        self.batch_file = self.batch_output_dir / f"batch_{batch_index:05d}.h5"
        self.batch_work_dir = self.scratch_root / f"batch_{batch_index:05d}"

        self.model_root = Path(config["paths"]["chemistry_model_root"]).resolve()
        self.compression = self.processing.get("compression", "gzip")
        self.compression_level = self.processing.get("compression_level", 4)
        self.cleanup_temporary = bool(self.processing.get("cleanup_temporary", True))
        self.reuse_trace_output = bool(self.processing.get("reuse_existing_trace_output", False))
        self.reuse_existing_ev_output = bool(self.processing.get("reuse_existing_ev_output", True))
        self._persisted_reuse_flag_changes: set[str] = set()

    def _persist_processing_flag(self, flag_name: str, value: bool, reason: str) -> None:
        """Persist one processing flag change to in-memory config and pipeline YAML."""
        if flag_name in self._persisted_reuse_flag_changes:
            return

        self.processing[flag_name] = bool(value)
        self._persisted_reuse_flag_changes.add(flag_name)
        print(f"Batch {self.batch_index}: setting processing.{flag_name}={value} ({reason})")

        config_path_raw = self.config.get("config_path")
        if not config_path_raw:
            return

        config_path = Path(config_path_raw)
        try:
            with open(config_path, "r", encoding="ascii") as handle:
                config_data = yaml.safe_load(handle)

            if not isinstance(config_data, dict):
                print(
                    f"Batch {self.batch_index}: warning: could not persist {flag_name}; config root is not a mapping"
                )
                return

            if "processing" not in config_data or not isinstance(config_data["processing"], dict):
                config_data["processing"] = {}

            config_data["processing"][flag_name] = bool(value)

            with open(config_path, "w", encoding="ascii") as handle:
                yaml.safe_dump(config_data, handle, sort_keys=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Batch {self.batch_index}: warning: failed to persist {flag_name} to config: {exc}")

    @staticmethod
    def _write_trace_config(config_path: Path, particle_ids: np.ndarray) -> None:
        """Write trace.cfg selecting only the batch particle ID range."""
        config_path.write_text(
            "&trace_config\n"
            f"  id_start = {int(particle_ids[0])}\n"
            f"  id_end = {int(particle_ids[-1])}\n"
            "  n_boundary = 0\n"
            "/\n",
            encoding="ascii",
        )

    @staticmethod
    def _ensure_link(source: Path, destination: Path) -> None:
        """Create or replace symlink destination -> source."""
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.symlink(source, destination)

    def _prepare_trace_workspace(self) -> Path:
        """Create batch-local trace workspace and return trace output directory."""
        phantomanalysis_binary = Path(self.config["paths"]["phantomanalysis_binary"]).resolve()
        av_dir = Path(self.config["paths"]["av_dir"]).resolve()

        FileSystemTools.ensure_clean_directory(self.batch_work_dir)
        trace_output_dir = FileSystemTools.ensure_clean_directory(self.batch_work_dir / "trace_output")
        self._write_trace_config(self.batch_work_dir / "trace.cfg", self.particle_ids)
        (self.batch_work_dir / "batch_particle_ids.json").write_text(
            json.dumps([int(pid) for pid in self.particle_ids.tolist()]), encoding="ascii"
        )

        copied_binary = self.batch_work_dir / "phantomanalysis"
        shutil.copy2(phantomanalysis_binary, copied_binary)
        copied_binary.chmod(0o755)
        self._ensure_link(av_dir, self.batch_work_dir / "AV")
        return trace_output_dir

    def _run_trace_stage(self) -> None:
        """Execute phantomanalysis across selected dumps to generate .phys files."""
        selected_dumps = DumpSelection.selected_dump_numbers(self.config)
        if len(selected_dumps) < 2:
            raise RuntimeError("Trace stage requires at least two dumps.")

        binary = self.batch_work_dir / "phantomanalysis"
        data_dir = Path(self.config["paths"]["phantom_dump_dir"])
        prefix = self.config["simulation"]["prefix"]
        dump_paths = [str(data_dir / f"{prefix}_{dump_number:05d}") for dump_number in selected_dumps]
        subprocess.run([str(binary), *dump_paths], cwd=self.batch_work_dir, check=True)

    def _all_trace_files_present(self, trace_output_dir: Path) -> bool:
        """Check whether trace_output already contains all expected particle files."""
        for particle_id in self.particle_ids.tolist():
            if not (trace_output_dir / f"{particle_id}.phys").exists():
                return False
        return True

    @staticmethod
    def _assert_fortran_path_length(value: str, label: str) -> None:
        """Validate path length against Fortran A70 reader limitation."""
        if len(value) > FORTRAN_PATH_LIMIT:
            raise ValueError(
                f"{label} path is too long for Fortran A70 reader ({len(value)} > {FORTRAN_PATH_LIMIT}): {value}"
            )

    def _chemistry_runtime_dirs(self, chemistry_type: str) -> dict[str, Path]:
        """Build per-chemistry runtime directory set under batch scratch area."""
        base = FileSystemTools.ensure_clean_directory(self.batch_work_dir / "chemistry_runtime" / chemistry_type)
        return {
            "base": base,
            "param": FileSystemTools.ensure_clean_directory(base / "param" / "trace_file_param"),
            "input": FileSystemTools.ensure_clean_directory(base / "input"),
            "output": FileSystemTools.ensure_clean_directory(base / "output"),
            "ev_output": FileSystemTools.ensure_clean_directory(base / "ev_output"),
            "analyse_output": FileSystemTools.ensure_clean_directory(base / "analyse_output"),
        }

    def _prepare_model_runtime_links(self, runtime_dirs: dict[str, Path], chemistry_type: str) -> Path:
        """Create short symlink-based runtime paths to avoid Fortran path overflows."""
        chem_tag = chemistry_type[:1].lower() if chemistry_type else "x"
        runtime_root = self.model_root / "_rt" / f"b{self.batch_index:05d}_{chem_tag}"
        FileSystemTools.ensure_clean_directory(runtime_root)

        self._ensure_link(runtime_dirs["input"], runtime_root / "in")
        self._ensure_link(runtime_dirs["output"], runtime_root / "out")
        self._ensure_link(runtime_dirs["ev_output"], runtime_root / "ev")
        self._ensure_link(runtime_dirs["analyse_output"], runtime_root / "ana")
        self._ensure_link(runtime_dirs["param"], runtime_root / "par")
        return runtime_root.relative_to(self.model_root)

    def _write_file_params(
        self,
        runtime_dirs: dict[str, Path],
        model_runtime_rel: Path,
        particle_id: int,
        chemistry_type: str,
    ) -> Path:
        """Write model file_parameters_*.txt for one particle and chemistry type."""
        file_params_path = runtime_dirs["param"] / f"file_parameters_{particle_id}.txt"

        phys_file = (model_runtime_rel / "in" / f"{particle_id}.txt").as_posix()
        reac_file = "rate12_complex.rates"
        spec_file = SpeciesCatalog.species_filename(chemistry_type)
        bind_file = "rate12_binding.dat"
        switch_file = "param/reaction_parameters.txt"
        grain_file = "param/grain_parameters.txt"
        rad_file = "param/radiation_parameters.txt"
        out_file = (model_runtime_rel / "out" / f"model_output_{particle_id}.dat").as_posix()
        rates_file = (model_runtime_rel / "out" / f"model_rates_{particle_id}.dat").as_posix()
        ana_file = (model_runtime_rel / "ana" / f"model_analyse_{particle_id}.dat").as_posix()

        for label, value in (
            ("physical", phys_file),
            ("reaction", reac_file),
            ("species", spec_file),
            ("binding", bind_file),
            ("switch", switch_file),
            ("grain", grain_file),
            ("radiation", rad_file),
            ("out", out_file),
            ("rates", rates_file),
            ("analyse", ana_file),
        ):
            self._assert_fortran_path_length(value, label)

        with open(file_params_path, "w", encoding="ascii") as handle:
            handle.write("! FILE PARAMETERS FOR ENVELOPE MODEL\n")
            handle.write("! Physical conditions input file\n")
            handle.write(f"{phys_file}\n")
            handle.write("! Reaction file\n")
            handle.write(f"{reac_file}\n")
            handle.write("! Species file\n")
            handle.write(f"{spec_file}\n")
            handle.write("! Binding energies file\n")
            handle.write(f"{bind_file}\n")
            handle.write("! Reaction parameters file\n")
            handle.write(f"{switch_file}\n")
            handle.write("! Grain parameters file\n")
            handle.write(f"{grain_file}\n")
            handle.write("! Radiation field parameters file\n")
            handle.write(f"{rad_file}\n")
            handle.write("! Output abundances file\n")
            handle.write(f"{out_file}\n")
            handle.write("! Output rates file\n")
            handle.write(f"{rates_file}\n")
            handle.write("! Call analyse subroutine (T(RUE) or F(ALSE))?\n")
            handle.write("F\n")
            handle.write("! Grid point to run analyse\n")
            handle.write("0\n")
            handle.write("! Analyse file\n")
            handle.write(f"{ana_file}\n")
        return file_params_path

    def _run_chemistry_model(self, file_params_arg: str) -> None:
        """Run compiled chemistry model binary for one particle parameters file."""
        self._assert_fortran_path_length(file_params_arg, "file_parameters")
        subprocess.run([str((self.model_root / "model").resolve()), file_params_arg], cwd=self.model_root, check=True)

    def _postprocess_ev_output(self, filepath: Path, to_cm: bool = True) -> None:
        """Ensure EV output includes R and converted position units in centimeters."""
        if not filepath.exists():
            return

        raw_lines = filepath.read_text(encoding="ascii").splitlines(keepends=True)
        if len(raw_lines) < 6:
            return

        names_idx = 4
        units_idx = 5

        col_names = raw_lines[names_idx].split()
        if "X" not in col_names or "Y" not in col_names or "Z" not in col_names:
            return

        xi = col_names.index("X")
        yi = col_names.index("Y")
        zi = col_names.index("Z")
        ri_existing = col_names.index("R") if "R" in col_names else -1

        pc_to_cm = 3.085677581491367e18
        unit_str = "(CM)" if to_cm else "(PC)"

        new_col_names = col_names[: zi + 1] + ["R"] + col_names[zi + 1 :] if ri_existing < 0 else col_names[:]
        names_line = "  ".join(f"{name:<16}" for name in new_col_names) + "\n"

        data_start = units_idx
        if units_idx < len(raw_lines) and raw_lines[units_idx].lstrip().startswith("#"):
            old_units = raw_lines[units_idx].lstrip("#").split()
            new_units = old_units[: zi + 1] + [unit_str] + old_units[zi + 1 :] if ri_existing < 0 else old_units[:]
            if to_cm:
                new_units[xi] = unit_str
                new_units[yi] = unit_str
                new_units[zi] = unit_str
                if ri_existing >= 0:
                    new_units[ri_existing] = unit_str
                else:
                    new_units[zi + 1] = unit_str
            units_line = "#  " + "  ".join(f"{unit:<16}" for unit in new_units) + "\n"
            data_start = units_idx + 1
        else:
            units_line = raw_lines[units_idx]

        new_lines = raw_lines[:names_idx] + [names_line, units_line]

        for line in raw_lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue

            parts = stripped.split()
            x = float(parts[xi])
            y = float(parts[yi])
            z = float(parts[zi])
            if to_cm:
                x *= pc_to_cm
                y *= pc_to_cm
                z *= pc_to_cm
                parts[xi] = f"{x:.5e}"
                parts[yi] = f"{y:.5e}"
                parts[zi] = f"{z:.5e}"

            r = np.sqrt(x * x + y * y + z * z)
            if ri_existing >= 0:
                parts[ri_existing] = f"{r:.5e}"
            else:
                parts = parts[: zi + 1] + [f"{r:.5e}"] + parts[zi + 1 :]

            new_lines.append("  ".join(parts) + "\n")

        filepath.write_text("".join(new_lines), encoding="ascii")

    def _convert_model_output(self, runtime_dirs: dict[str, Path], particle_id: int, chemistry_type: str) -> Path:
        """Convert model output to EV table and apply post-processing fixes."""
        output_prefix = runtime_dirs["ev_output"] / f"ev_{particle_id}"
        command = [
            "perl",
            str((self.model_root / "evolve_output.pl").resolve()),
            str((runtime_dirs["output"] / f"model_output_{particle_id}.dat").resolve()),
            str(output_prefix.resolve()),
        ]
        command.extend(SpeciesCatalog.list_for_chemistry(chemistry_type))
        subprocess.run(command, cwd=self.model_root, check=True)

        ev_output_path = output_prefix.with_suffix(".dat")
        self._postprocess_ev_output(ev_output_path, to_cm=True)
        return ev_output_path

    @staticmethod
    def _cleanup_particle_products(
        trace_file: Path | None,
        runtime_dirs: dict[str, Path],
        particle_id: int,
        cleanup_trace_file: bool,
        cleanup_ev_output: bool,
    ) -> None:
        """Remove per-particle temporary files once values are written to HDF5."""
        candidates = [
            runtime_dirs["input"] / f"{particle_id}.txt",
            runtime_dirs["output"] / f"model_output_{particle_id}.dat",
            runtime_dirs["output"] / f"model_rates_{particle_id}.dat",
            runtime_dirs["analyse_output"] / f"model_analyse_{particle_id}.dat",
            runtime_dirs["param"] / f"file_parameters_{particle_id}.txt",
        ]
        if cleanup_ev_output:
            candidates.append(runtime_dirs["ev_output"] / f"ev_{particle_id}.dat")
        if cleanup_trace_file and trace_file is not None:
            candidates.append(trace_file)

        for candidate in candidates:
            if candidate.exists():
                candidate.unlink()

    def _prepare_batch_paths(self) -> None:
        """Apply overwrite/reuse policy before running expensive stages."""
        overwrite_batch = bool(self.processing.get("overwrite_batch", False))
        if self.batch_file.exists():
            if not overwrite_batch:
                raise FileExistsError(f"Batch output already exists: {self.batch_file}")
            self.batch_file.unlink()

        if self.batch_work_dir.exists() and not self.reuse_trace_output:
            shutil.rmtree(self.batch_work_dir)

    def run(self) -> Path:
        """Execute complete batch workflow and return generated batch file path."""
        self._prepare_batch_paths()

        print(
            f"Batch {self.batch_index}: particles {self.metadata['selection_start']}:{self.metadata['selection_end']} "
            f"({len(self.particle_ids)} particle(s) out of {self.metadata['total_particles']})"
        )

        trace_output_dir = self._prepare_trace_workspace()
        if self.reuse_trace_output and self._all_trace_files_present(trace_output_dir):
            print(f"Batch {self.batch_index}: reusing existing trace_output (skip phantomanalysis)")
        else:
            if self.reuse_trace_output:
                self._persist_processing_flag(
                    "reuse_existing_trace_output",
                    False,
                    "trace_output is incomplete for this batch",
                )
                self.reuse_trace_output = False
            self._run_trace_stage()

        BatchHDF5Appender.initialise_batch_file(
            batch_file=self.batch_file,
            batch_index=self.batch_index,
            particle_ids=self.particle_ids,
            dump_numbers=self.dump_numbers,
            chemistry_types=self.chemistry_types,
        )

        runtime_dirs_by_type: dict[str, dict[str, Path]] = {}
        runtime_rel_by_type: dict[str, Path] = {}
        for chemistry_type in self.chemistry_types:
            runtime_dirs = self._chemistry_runtime_dirs(chemistry_type)
            runtime_dirs_by_type[chemistry_type] = runtime_dirs
            runtime_rel_by_type[chemistry_type] = self._prepare_model_runtime_links(runtime_dirs, chemistry_type)

        for slot, particle_id in enumerate(self.particle_ids.tolist()):
            trace_file = trace_output_dir / f"{particle_id}.phys"
            if not trace_file.exists():
                raise FileNotFoundError(f"Missing trace file for particle {particle_id}: {trace_file}")

            for chemistry_type in self.chemistry_types:
                runtime_dirs = runtime_dirs_by_type[chemistry_type]
                model_runtime_rel = runtime_rel_by_type[chemistry_type]
                input_file = runtime_dirs["input"] / f"{particle_id}.txt"
                ev_output_path = runtime_dirs["ev_output"] / f"ev_{particle_id}.dat"

                if self.reuse_existing_ev_output and ev_output_path.exists() and ev_output_path.stat().st_size == 0:
                    self._persist_processing_flag(
                        "reuse_existing_ev_output",
                        False,
                        f"found empty ev_output file: {ev_output_path.name}",
                    )
                    self.reuse_existing_ev_output = False

                if self.reuse_existing_ev_output and ev_output_path.exists() and ev_output_path.stat().st_size > 0:
                    print(
                        f"Batch {self.batch_index}: reusing existing ev_output for particle {particle_id} ({chemistry_type})"
                    )
                else:
                    PhysTraceConverter.convert_phys_file(trace_file, input_file)
                    file_params_path = self._write_file_params(runtime_dirs, model_runtime_rel, particle_id, chemistry_type)
                    file_params_arg = (model_runtime_rel / "par" / file_params_path.name).as_posix()
                    self._run_chemistry_model(file_params_arg)
                    ev_output_path = self._convert_model_output(runtime_dirs, particle_id, chemistry_type)

                BatchHDF5Appender.append_particle_to_batch(
                    batch_file=self.batch_file,
                    slot=slot,
                    particle_id=particle_id,
                    chemistry_type=chemistry_type,
                    ev_output_path=ev_output_path,
                    dump_numbers=self.dump_numbers,
                    compression=self.compression,
                    compression_level=self.compression_level,
                )

                self._cleanup_particle_products(
                    trace_file,
                    runtime_dirs,
                    particle_id,
                    cleanup_trace_file=False,
                    cleanup_ev_output=not self.reuse_existing_ev_output,
                )

            if self.cleanup_temporary and trace_file.exists():
                trace_file.unlink()

        if self.cleanup_temporary:
            for model_runtime_rel in runtime_rel_by_type.values():
                link_root = self.model_root / model_runtime_rel
                if link_root.exists():
                    shutil.rmtree(link_root)
            shutil.rmtree(self.batch_work_dir)

        return self.batch_file


class BatchPipelineCLI:
    """CLI wrapper for single-batch execution."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse required config path and batch index."""
        parser = argparse.ArgumentParser(description="Run one batch of the standalone HDF5 chemistry pipeline.")
        parser.add_argument("--config", required=True)
        parser.add_argument("--batch-index", type=int, required=True)
        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        """Execute batch pipeline and print completion summary."""
        args = cls.parse_args()
        config = PipelineConfigManager.load(args.config)
        batch_file = BatchPipelineRunner(config=config, batch_index=args.batch_index).run()
        print(f"Completed batch {args.batch_index}: {batch_file}")


def main() -> None:
    """CLI entrypoint."""
    BatchPipelineCLI.run()


if __name__ == "__main__":
    main()
