from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import FileSystemTools, PipelineConfigManager


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
    def _collect_dump_data(
        batch_files: list[Path],
        dump_number: int,
        chemistry_types: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        """Collect one dump's trace and chemistry arrays across all batch files."""
        trace_values: dict[str, list[np.ndarray]] = {"id": []}
        chemistry_values: dict[str, dict[str, list[np.ndarray]]] = {chemistry_type: {} for chemistry_type in chemistry_types}

        for batch_file in batch_files:
            with h5py.File(batch_file, "r") as handle:
                particle_ids = handle["particles/id"][:]
                valid_mask = particle_ids > 0
                if not np.any(valid_mask):
                    continue

                dump_numbers = [int(value) for value in json.loads(handle.attrs["dump_numbers_json"])]
                if dump_number not in dump_numbers:
                    continue
                row_index = dump_numbers.index(dump_number)

                if "particles" in handle["trace"]:
                    trace_particles = handle["trace/particles"]
                    row_count = trace_particles["row_count"][:]
                    include_indices = np.where(valid_mask & (row_count > row_index))[0]
                    if include_indices.size == 0:
                        continue

                    trace_values["id"].append(particle_ids[include_indices].astype(np.int64))

                    for dataset_name in trace_particles.keys():
                        if dataset_name == "row_count":
                            continue
                        dataset = trace_particles[dataset_name]
                        values = np.asarray(
                            [np.float64(dataset[idx][row_index]) for idx in include_indices], dtype=np.float64
                        )
                        trace_values.setdefault(dataset_name, []).append(values)

                    for chemistry_type in chemistry_types:
                        chemistry_particles = handle[f"chemistry/{chemistry_type}/particles"]
                        chem_row_count = chemistry_particles["row_count"][:]
                        chem_include = np.where(valid_mask & (chem_row_count > row_index))[0]
                        if chem_include.size == 0:
                            continue
                        for dataset_name in chemistry_particles.keys():
                            if dataset_name == "row_count":
                                continue
                            dataset = chemistry_particles[dataset_name]
                            values = np.asarray(
                                [np.float64(dataset[idx][row_index]) for idx in chem_include], dtype=np.float64
                            )
                            chemistry_values[chemistry_type].setdefault(dataset_name, []).append(values)
                else:
                    # Legacy fallback retained to preserve support for older batch layout.
                    trace_group = handle[f"trace/dumps/dump_{dump_number:05d}"]
                    trace_values["id"].append(particle_ids[valid_mask].astype(np.int64))
                    for dataset_name in trace_group.keys():
                        trace_values.setdefault(dataset_name, []).append(
                            trace_group[dataset_name][:][valid_mask].astype(np.float64)
                        )

                    for chemistry_type in chemistry_types:
                        chemistry_group = handle[f"chemistry/{chemistry_type}/dumps/dump_{dump_number:05d}"]
                        for dataset_name in chemistry_group.keys():
                            chemistry_values[chemistry_type].setdefault(dataset_name, []).append(
                                chemistry_group[dataset_name][:][valid_mask].astype(np.float64)
                            )

        merged_trace = {
            name: np.concatenate(values) if values else np.array([], dtype=np.float64)
            for name, values in trace_values.items()
        }
        merged_trace["id"] = (
            np.concatenate(trace_values["id"]).astype(np.int64) if trace_values["id"] else np.array([], dtype=np.int64)
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

    def merge(self) -> list[Path]:
        """Merge all batches and return output dump file paths."""
        batch_files = self.list_batch_files()
        dump_numbers, chemistry_types = self.first_batch_metadata(batch_files)
        output_files = []

        for dump_number in dump_numbers:
            trace_data, chemistry_data = self._collect_dump_data(batch_files, dump_number, chemistry_types)
            output_path = self.final_output_dir / f"dump_{dump_number:05d}.h5"
            self._write_dump_file(
                output_path=output_path,
                dump_number=dump_number,
                trace_data=trace_data,
                chemistry_data=chemistry_data,
                chemistry_types=chemistry_types,
            )
            output_files.append(output_path)

        if not bool(self.config["processing"].get("retain_batch_files_after_merge", True)):
            for batch_file in batch_files:
                batch_file.unlink()

        return output_files


class MergeCLI:
    """CLI front-end for BatchFileMerger."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse required config argument."""
        parser = argparse.ArgumentParser(description="Merge batch_*.h5 files into one dump_XXXXX.h5 file per dump.")
        parser.add_argument("--config", required=True)
        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        """Run merge operation and print output count."""
        args = cls.parse_args()
        config = PipelineConfigManager.load(args.config)
        merger = BatchFileMerger(config)
        output_files = merger.merge()
        print(f"Merged {len(output_files)} dump file(s)")


def main() -> None:
    """CLI entrypoint."""
    MergeCLI.run()


if __name__ == "__main__":
    main()
