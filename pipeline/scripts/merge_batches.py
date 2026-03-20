from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import DumpTimeMapper, FileSystemTools, PipelineConfigManager


class BatchFileMerger:
    """Merge batch_*.h5 intermediate files into per-dump merged outputs."""

    def __init__(self, config: dict) -> None:
        """Capture config and commonly-used output options."""
        self.config = config
        self.batch_output_dir = Path(config["paths"]["batch_output_dir"]).resolve()
        self.final_output_dir = FileSystemTools.ensure_clean_directory(config["paths"]["final_output_dir"])
        self.compression = config["processing"].get("compression", "gzip")
        self.compression_level = config["processing"].get("compression_level", 4)

    def list_batch_files(self) -> list[Path]:
        """Return sorted batch files or fail early if none are available."""
        files = sorted(self.batch_output_dir.glob("batch_*.h5"))
        if not files:
            raise FileNotFoundError(f"No batch files found in {self.batch_output_dir}")
        return files

    @staticmethod
    def first_batch_metadata(batch_files: list[Path]) -> tuple[list[int], list[str]]:
        """Read global dump/chemistry metadata from first batch file."""
        with h5py.File(batch_files[0], "r") as handle:
            dump_numbers = [int(value) for value in json.loads(handle.attrs["dump_numbers_json"])]
            chemistry_types = list(json.loads(handle.attrs["chemistry_types_json"]))
        return dump_numbers, chemistry_types


    @staticmethod
    def _dataset_write_kwargs(compression: str | None, compression_level: int | None) -> dict:
        """Build h5py dataset creation kwargs from compression settings."""
        kwargs = {}
        if compression:
            kwargs["compression"] = compression
            if compression == "gzip" and compression_level is not None:
                kwargs["compression_opts"] = int(compression_level)
        return kwargs


    @staticmethod
    def _assert_single_dump_time(dump_number: int, trace_data: dict[str, np.ndarray]) -> None:
        """Validate that merged dump contains one unique time value across particles."""
        if "time" not in trace_data:
            return

        time_values = np.asarray(trace_data["time"], dtype=np.float64)
        if time_values.size == 0:
            return

        # Round away tiny floating-point print/noise differences before uniqueness check.
        rounded = np.round(time_values, decimals=10)
        unique = np.unique(rounded)
        if unique.size > 1:
            raise ValueError(
                "Merged dump has inconsistent particle times for "
                f"dump {dump_number:05d}: min={np.min(time_values):.6g}, "
                f"max={np.max(time_values):.6g}, unique_count={unique.size}"
            )

    @staticmethod
    def _collect_dump_data(
        batch_files: list[Path],
        dump_number: int,
        chemistry_types: list[str],
        canonical_dump_seconds: dict[int, float] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        """Collect one dump's data using fast vectorized column reads."""

        trace_values: dict[str, list[np.ndarray]] = {"id": []}
        chemistry_values: dict[str, dict[str, list[np.ndarray]]] = {
            chemistry_type: {} for chemistry_type in chemistry_types
        }

        for batch_file in batch_files:
            with h5py.File(batch_file, "r") as handle:
                particle_ids = handle["particles/id"][:]
                valid_mask = particle_ids > 0
                if not np.any(valid_mask):
                    continue

                dump_numbers = [int(value) for value in json.loads(handle.attrs["dump_numbers_json"])]
                if dump_number not in dump_numbers:
                    continue

                t_index = dump_numbers.index(dump_number)

                valid_ids = particle_ids[valid_mask]

                trace_particles = handle["trace/particles"]

                # Use a reference dataset to determine which particles exist at this dump
                reference_name = next(iter(trace_particles.keys()))
                reference_column = trace_particles[reference_name][:, t_index]
                reference_values = reference_column[valid_mask]

                present_mask = ~np.isnan(reference_values)
                ids = valid_ids[present_mask]

                # Store IDs aligned with actual data presence
                trace_values["id"].append(ids.astype(np.int64))

                for dataset_name in trace_particles.keys():
                    dataset = trace_particles[dataset_name]

                    # Vectorized column read
                    column = dataset[:, t_index]

                    values = column[valid_mask]

                    # Apply the SAME mask as used for IDs (ensures alignment)
                    values = values[present_mask]

                    if dataset_name == "time":
                        if canonical_dump_seconds is not None:
                            values = np.full(values.shape[0], float(canonical_dump_seconds[int(dump_number)]))
                    trace_values.setdefault(dataset_name, []).append(values)

                for chemistry_type in chemistry_types:
                    chemistry_particles = handle[f"chemistry/{chemistry_type}/particles"]

                    for dataset_name in chemistry_particles.keys():
                        dataset = chemistry_particles[dataset_name]

                        column = dataset[:, t_index]
                        values = column[valid_mask]

                        present_mask = ~np.isnan(values)
                        values = values[present_mask]

                        chemistry_values[chemistry_type].setdefault(dataset_name, []).append(values)

        merged_trace = {
            name: np.concatenate(values) if values else np.array([], dtype=np.float64)
            for name, values in trace_values.items()
        }
        merged_trace["id"] = (
            np.concatenate(trace_values["id"]).astype(np.int64)
            if trace_values["id"]
            else np.array([], dtype=np.int64)
        )

        merged_chemistry: dict[str, dict[str, np.ndarray]] = {}
        for chemistry_type, species_map in chemistry_values.items():
            merged_chemistry[chemistry_type] = {
                name: np.concatenate(values) if values else np.array([], dtype=np.float64)
                for name, values in species_map.items()
            }

        return merged_trace, merged_chemistry

    def _write_dump_file(
        self,
        output_path: Path,
        dump_number: int,
        trace_data: dict[str, np.ndarray],
        chemistry_data: dict[str, dict[str, np.ndarray]],
        chemistry_types: list[str],
    ) -> None:
        """Write one merged dump file using compact per-particle flat arrays."""
        kwargs = self._dataset_write_kwargs(self.compression, self.compression_level)

        with h5py.File(output_path, "w") as handle:
            handle.attrs["dump_number"] = int(dump_number)
            handle.attrs["n_particles"] = int(trace_data["id"].shape[0])
            handle.attrs["chemistry_types_json"] = json.dumps(chemistry_types)

            particles = handle.create_group("particles")
            particles.create_dataset("id", data=trace_data["id"], dtype=np.int64)

            for dataset_name, values in trace_data.items():
                if dataset_name == "id":
                    continue
                particles.create_dataset(dataset_name, data=values.astype(np.float64), dtype=np.float64, **kwargs)

            if len(chemistry_types) == 1:
                chemistry_type = chemistry_types[0]
                for dataset_name, values in chemistry_data[chemistry_type].items():
                    particles.create_dataset(dataset_name, data=values.astype(np.float64), dtype=np.float64, **kwargs)
            else:
                chemistry_group = handle.create_group("chemistry")
                for chemistry_type in chemistry_types:
                    chemistry_particles = chemistry_group.create_group(chemistry_type).create_group("particles")
                    for dataset_name, values in chemistry_data[chemistry_type].items():
                        chemistry_particles.create_dataset(
                            dataset_name,
                            data=values.astype(np.float64),
                            dtype=np.float64,
                            **kwargs,
                        )

    def merge_one(self, dump_index: int) -> Path:
        """Merge all batches for a single dump by its 0-based index and return the output path.

        This is the entry point used by the parallel SLURM array job, where each task
        receives SLURM_ARRAY_TASK_ID as dump_index.
        """
        batch_files = self.list_batch_files()
        dump_numbers, chemistry_types = self.first_batch_metadata(batch_files)

        if dump_index < 0 or dump_index >= len(dump_numbers):
            raise IndexError(
                f"dump_index {dump_index} is out of range for {len(dump_numbers)} dump(s)"
            )

        dump_number = dump_numbers[dump_index]
        canonical_times = DumpTimeMapper.load_or_build_dump_time_seconds(self.config, dump_numbers)
        canonical_dump_seconds = {int(d): float(s) for d, s in zip(dump_numbers, canonical_times)}

        trace_data, chemistry_data = self._collect_dump_data(
            batch_files,
            dump_number,
            chemistry_types,
            canonical_dump_seconds=canonical_dump_seconds,
        )
        self._assert_single_dump_time(dump_number=dump_number, trace_data=trace_data)
        output_path = self.final_output_dir / f"dump_{dump_number:05d}.h5"
        self._write_dump_file(
            output_path=output_path,
            dump_number=dump_number,
            trace_data=trace_data,
            chemistry_data=chemistry_data,
            chemistry_types=chemistry_types,
        )
        return output_path

    def merge(self) -> list[Path]:
        """Merge all batches serially and return output dump file paths.

        Kept for backward compatibility and standalone / debugging use.
        The normal pipeline path uses merge_one() via a SLURM array job.
        """
        batch_files = self.list_batch_files()
        dump_numbers, chemistry_types = self.first_batch_metadata(batch_files)
        output_files = [self.merge_one(i) for i in range(len(dump_numbers))]

        if not bool(self.config["processing"].get("retain_batch_files_after_merge", True)):
            for batch_file in batch_files:
                batch_file.unlink()

        return output_files


class MergeCLI:
    """CLI front-end for BatchFileMerger."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse required config argument and optional dump index."""
        parser = argparse.ArgumentParser(description="Merge batch_*.h5 files into one dump_XXXXX.h5 file per dump.")
        parser.add_argument("--config", required=True)
        parser.add_argument(
            "--dump-index",
            type=int,
            default=None,
            help=(
                "0-based index into the dump list to process. "
                "When set, only that one dump is merged (used by the SLURM array job, "
                "where SLURM_ARRAY_TASK_ID is passed as this value). "
                "Omit to run all dumps serially."
            ),
        )
        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        """Run merge operation: one dump if --dump-index is given, all dumps otherwise."""
        args = cls.parse_args()
        config = PipelineConfigManager.load(args.config)
        merger = BatchFileMerger(config)
        if args.dump_index is not None:
            output_file = merger.merge_one(args.dump_index)
            print(f"Merged dump file: {output_file}")
        else:
            output_files = merger.merge()
            print(f"Merged {len(output_files)} dump file(s)")


def main() -> None:
    """CLI entrypoint."""
    MergeCLI.run()


if __name__ == "__main__":
    main()