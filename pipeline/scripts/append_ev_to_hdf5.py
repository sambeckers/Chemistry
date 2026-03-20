from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np

from common import DumpTimeMapper, SpeciesCatalog, TRACE_COLUMN_MAP


class BatchHDF5Appender:
    """Create and append per-particle chemistry outputs into batch HDF5 files."""

    @staticmethod
    def parse_ev_output(ev_output_path: str | Path) -> tuple[list[str], np.ndarray]:
        """Parse evolve_output.pl text output into (column_names, numeric matrix)."""
        ev_output_path = Path(ev_output_path)
        lines = ev_output_path.read_text(encoding="ascii").splitlines()

        # The output can contain descriptive preamble lines. We identify the actual data
        # header by looking for the canonical coordinate/state fields.
        header_index = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("X") and "DENSITY" in stripped and "TIME" in stripped:
                header_index = idx
                break
        if header_index is None:
            raise ValueError(f"Could not locate data header in {ev_output_path}")

        column_names = lines[header_index].split()
        data_lines = []
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data_lines.append(stripped)
        if not data_lines:
            raise ValueError(f"No data rows found in {ev_output_path}")

        data = np.loadtxt(io.StringIO("\n".join(data_lines)), dtype=np.float64, ndmin=2)
        if data.shape[1] != len(column_names):
            raise ValueError(
                f"Column count mismatch in {ev_output_path}: header has {len(column_names)}, data has {data.shape[1]}"
            )
        return column_names, data

    @staticmethod
    def aligned_dump_numbers_for_rows(dump_numbers: list[int], row_count: int) -> np.ndarray:
        """Return the ordered dump-number suffix represented by one particle trace."""
        if row_count > len(dump_numbers):
            raise ValueError(f"Row count {row_count} exceeds configured dump count {len(dump_numbers)}")
        start_index = len(dump_numbers) - int(row_count)
        return np.asarray(dump_numbers[start_index:], dtype=np.int32)

    @staticmethod
    def initialise_batch_file(
        batch_file: str | Path,
        batch_index: int,
        particle_ids: np.ndarray,
        dump_numbers: list[int],
        chemistry_types: list[str],
        dump_time_seconds: list[float] | None = None,
        time_key_decimals: int = 0,
    ) -> None:
        """Create an empty batch file with metadata and fixed bookkeeping datasets."""
        batch_file = Path(batch_file)
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(batch_file, "w") as handle:
            handle.attrs["batch_index"] = int(batch_index)
            handle.attrs["n_particles"] = int(len(particle_ids))
            handle.attrs["dump_numbers_json"] = json.dumps([int(dump) for dump in dump_numbers])
            if dump_time_seconds is not None:
                handle.attrs["dump_time_seconds_json"] = json.dumps([float(value) for value in dump_time_seconds])
            handle.attrs["time_key_decimals"] = int(time_key_decimals)
            handle.attrs["chemistry_types_json"] = json.dumps(list(chemistry_types))

            particles_group = handle.create_group("particles")
            particles_group.create_dataset("expected_id", data=np.asarray(particle_ids, dtype=np.int64))
            particles_group.create_dataset("id", shape=(len(particle_ids),), dtype=np.int64, fillvalue=-1)

            n_particles = len(particle_ids)
            n_dumps = len(dump_numbers)
            trace_particles = handle.create_group("trace").create_group("particles")
            trace_particles.create_dataset("density", shape=(n_particles, n_dumps), dtype=np.float64, fillvalue=np.nan)
            trace_particles.create_dataset("temperature", shape=(n_particles, n_dumps), dtype=np.float64, fillvalue=np.nan)
            trace_particles.create_dataset("time", shape=(n_particles, n_dumps), dtype=np.float64, fillvalue=np.nan)
            trace_particles.create_dataset("r", shape=(n_particles, n_dumps), dtype=np.float64, fillvalue=np.nan)

            chemistry_group = handle.create_group("chemistry")
            for chemistry_type in chemistry_types:
                chemistry_particles = chemistry_group.create_group(chemistry_type).create_group("particles")
                chemistry_particles.create_dataset("placeholder", shape=(n_particles, n_dumps), dtype=np.float64, fillvalue=np.nan)

    @classmethod
    def append_particle_to_batch(
        cls,
        batch_file: str | Path,
        slot: int,
        particle_id: int,
        chemistry_type: str,
        ev_output_path: str | Path,
        dump_numbers: list[int],
        dump_time_seconds: list[float] | None = None,
        time_key_decimals: int | None = None,
        compression: str | None = "gzip",
        compression_level: int | None = 4,
    ) -> None:
        """Append one particle evolution table into the given batch file slot."""
        del compression
        del compression_level

        column_names, data = cls.parse_ev_output(ev_output_path)
        if data.shape[0] > len(dump_numbers):
            raise ValueError(
                f"{ev_output_path} contains {data.shape[0]} rows but batch expects at most {len(dump_numbers)} dumps"
            )

        column_data = {name: data[:, index] for index, name in enumerate(column_names)}
        if "R" not in column_data:
            if not all(key in column_data for key in ("X", "Y", "Z")):
                raise ValueError(f"{ev_output_path} has no R column and cannot derive it (missing X/Y/Z)")
            column_data["R"] = np.sqrt(
                np.asarray(column_data["X"], dtype=np.float64) ** 2
                + np.asarray(column_data["Y"], dtype=np.float64) ** 2
                + np.asarray(column_data["Z"], dtype=np.float64) ** 2
            )

        with h5py.File(batch_file, "a") as handle:
            particle_ids = handle["particles/id"]
            n_particles = int(particle_ids.shape[0])
            if slot < 0 or slot >= n_particles:
                raise IndexError(f"Slot {slot} is out of range for {n_particles} particle(s)")
            particle_ids[slot] = int(particle_id)

            n_rows = int(data.shape[0])

            chemistry_group = handle[f"chemistry/{chemistry_type}/particles"]
            name_map = json.loads(chemistry_group.attrs.get("species_name_map_json", "{}"))

            if dump_time_seconds is None:
                dump_time_seconds_json = handle.attrs.get("dump_time_seconds_json")
                if dump_time_seconds_json:
                    dump_time_seconds = [float(value) for value in json.loads(dump_time_seconds_json)]

            if time_key_decimals is None:
                time_key_decimals = int(handle.attrs.get("time_key_decimals", 0))

            mapped_dump_numbers: np.ndarray | None = None
            if dump_time_seconds is not None:
                if "TIME" in column_data:
                    try:
                        # EV TIME is written in years by evolve_output.pl.
                        mapped_dump_numbers = DumpTimeMapper.map_time_values_to_dump_numbers(
                            time_values=np.asarray(column_data["TIME"], dtype=np.float64),
                            dump_numbers=dump_numbers,
                            dump_time_seconds=dump_time_seconds,
                            decimals=int(time_key_decimals),
                            unit_hint="years",
                        )
                    except ValueError:
                        # Fallback for legacy chemistry outputs where TIME precision is too coarse.
                        mapped_dump_numbers = cls.aligned_dump_numbers_for_rows(
                            dump_numbers=dump_numbers,
                            row_count=n_rows,
                        )
                else:
                    mapped_dump_numbers = cls.aligned_dump_numbers_for_rows(
                        dump_numbers=dump_numbers,
                        row_count=n_rows,
                    )

            if mapped_dump_numbers is not None:
                dump_index_map = {dump: i for i, dump in enumerate(dump_numbers)}
                indices = np.array([dump_index_map[int(d)] for d in mapped_dump_numbers], dtype=int)

                dump_to_seconds = {
                    int(dump): float(seconds) for dump, seconds in zip(dump_numbers, dump_time_seconds)
                }
                # Canonicalise stored TIME values to exact dump-time map seconds.
                column_data["TIME"] = np.asarray(
                    [dump_to_seconds[int(dump)] for dump in mapped_dump_numbers], dtype=np.float64
                )

                trace_particles = handle["trace/particles"]
                for source_name, dataset_name in TRACE_COLUMN_MAP.items():
                    if dataset_name not in trace_particles:
                        trace_particles.create_dataset(
                            dataset_name,
                            shape=(n_particles, len(dump_numbers)),
                            dtype=np.float64,
                            fillvalue=np.nan,
                        )
                    trace_particles[dataset_name][slot, indices] = np.asarray(column_data[source_name], dtype=np.float64)

                chemistry_particles = handle[f"chemistry/{chemistry_type}/particles"]
                species_names = [name for name in column_names if name not in TRACE_COLUMN_MAP]
                for source_name in species_names:
                    dataset_name = SpeciesCatalog.normalise_name(source_name)
                    if dataset_name not in chemistry_particles:
                        chemistry_particles.create_dataset(
                            dataset_name,
                            shape=(n_particles, len(dump_numbers)),
                            dtype=np.float64,
                            fillvalue=np.nan,
                        )
                    chemistry_particles[dataset_name][slot, indices] = np.asarray(column_data[source_name], dtype=np.float64)

                chemistry_group.attrs["species_name_map_json"] = json.dumps(name_map, sort_keys=True)


class BatchHDF5CLI:
    """CLI wrapper for init/append commands."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command-line arguments for init/append subcommands."""
        parser = argparse.ArgumentParser(
            description="Initialise or append chemistry evolution outputs into a batch HDF5 file."
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        init_parser = subparsers.add_parser("init")
        init_parser.add_argument("batch_file")
        init_parser.add_argument("batch_index", type=int)
        init_parser.add_argument("particle_ids_json")
        init_parser.add_argument("dump_numbers_json")
        init_parser.add_argument("dump_time_seconds_json")
        init_parser.add_argument("chemistry_types_json")
        init_parser.add_argument("--time-key-decimals", type=int, default=0)

        append_parser = subparsers.add_parser("append")
        append_parser.add_argument("batch_file")
        append_parser.add_argument("slot", type=int)
        append_parser.add_argument("particle_id", type=int)
        append_parser.add_argument("chemistry_type")
        append_parser.add_argument("ev_output_path")
        append_parser.add_argument("dump_numbers_json")
        append_parser.add_argument("dump_time_seconds_json")
        append_parser.add_argument("--time-key-decimals", type=int, default=0)
        append_parser.add_argument("--compression", default="gzip")
        append_parser.add_argument("--compression-level", type=int, default=4)

        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        """Execute selected subcommand with backward-compatible semantics."""
        args = cls.parse_args()
        if args.command == "init":
            BatchHDF5Appender.initialise_batch_file(
                batch_file=args.batch_file,
                batch_index=args.batch_index,
                particle_ids=np.asarray(json.loads(args.particle_ids_json), dtype=np.int64),
                dump_numbers=[int(value) for value in json.loads(args.dump_numbers_json)],
                dump_time_seconds=[float(value) for value in json.loads(args.dump_time_seconds_json)],
                time_key_decimals=int(args.time_key_decimals),
                chemistry_types=list(json.loads(args.chemistry_types_json)),
            )
            return

        BatchHDF5Appender.append_particle_to_batch(
            batch_file=args.batch_file,
            slot=args.slot,
            particle_id=args.particle_id,
            chemistry_type=args.chemistry_type,
            ev_output_path=args.ev_output_path,
            dump_numbers=[int(value) for value in json.loads(args.dump_numbers_json)],
            dump_time_seconds=[float(value) for value in json.loads(args.dump_time_seconds_json)],
            time_key_decimals=int(args.time_key_decimals),
            compression=args.compression,
            compression_level=args.compression_level,
        )


def main() -> None:
    """CLI entrypoint."""
    BatchHDF5CLI.run()


if __name__ == "__main__":
    main()
