"""
gather_dumps.py — Gather phase of the two-stage merge.

Each SLURM array task (0 … n_dumps-1) opens every scatter_*.h5 file but
reads ONLY the group for its target dump number.  Because the scatter phase
already extracted and stored per-dump slices as contiguous HDF5 datasets,
each file open involves reading only ~n_particles_per_batch × n_datasets
bytes rather than the entire batch.

Fast-skip optimisation: the scatter files store a ``dumps_present_json``
root attribute listing which dump numbers have at least one particle.  The
gather task checks this attribute before touching any group structure, so
files with no contribution to the target dump are skipped after a single
small attribute read.

Scatter file format expected (written by scatter_batches.py):

    scatter_NNNNN.h5
      attrs:
        dump_numbers_json    "[0, 1, 2, …]"
        chemistry_types_json '["Crich"]'
        dumps_present_json   "[3, 7, 12, …]"
        trace_fields_json    '["x", "y", "z", "temp", …]'
        species_json         '{"Crich": ["c", "c2", …]}'
      /dump_00003/
        attrs: dump_number, n_particles
        id            (int64,   1-D: n_particles)
        trace         (float64, 2-D: n_particles × n_trace_fields)
        Crich/
          abundances  (float64, 2-D: n_particles × n_species)

The final output format is identical to the original merge_batches.py output,
ensuring downstream consumers see no difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import DumpTimeMapper, FileSystemTools, PipelineConfigManager


class ScatterGatherer:
    """Assemble per-dump final files from scatter_*.h5 contributions."""

    def __init__(self, config: dict) -> None:
        self.config = config
        scatter_path = config["paths"].get(
            "scatter_output_dir",
            str(Path(config["paths"]["batch_output_dir"]).parent / "scatter"),
        )
        self.scatter_dir = Path(scatter_path).resolve()
        self.final_output_dir = FileSystemTools.ensure_clean_directory(
            config["paths"]["final_output_dir"]
        )
        self.compression = config["processing"].get("compression", "gzip")
        self.compression_level = config["processing"].get("compression_level", 4)

    # ------------------------------------------------------------------ helpers

    def list_scatter_files(self) -> list[Path]:
        files = sorted(self.scatter_dir.glob("scatter_*.h5"))
        if not files:
            raise FileNotFoundError(f"No scatter files found in {self.scatter_dir}")
        return files

    @staticmethod
    def _first_file_metadata(
        scatter_files: list[Path],
    ) -> tuple[list[int], list[str], list[str], dict[str, list[str]]]:
        """
        Read global metadata from the first scatter file.

        Returns
        -------
        dump_numbers    : ordered list of all dump numbers in the run
        chemistry_types : e.g. ["Crich"]
        trace_fields    : ordered list of trace field names
        species_names   : {ctype: [species, …]} mapping
        """
        with h5py.File(scatter_files[0], "r") as f:
            dump_numbers = [int(v) for v in json.loads(f.attrs["dump_numbers_json"])]
            chemistry_types = list(json.loads(f.attrs["chemistry_types_json"]))
            trace_fields = list(json.loads(f.attrs["trace_fields_json"]))
            species_names: dict[str, list[str]] = json.loads(f.attrs["species_json"])
        return dump_numbers, chemistry_types, trace_fields, species_names

    @staticmethod
    def _write_kwargs(compression: str | None, level: int | None) -> dict:
        kw: dict = {}
        if compression:
            kw["compression"] = compression
            if compression == "gzip" and level is not None:
                kw["compression_opts"] = int(level)
        return kw

    @staticmethod
    def _assert_single_dump_time(dump_number: int, times: np.ndarray) -> None:
        """Validate that all particles in the merged dump share one time value."""
        if times.size == 0:
            return
        rounded = np.round(times, decimals=10)
        unique = np.unique(rounded)
        if unique.size > 1:
            raise ValueError(
                f"Inconsistent particle times for dump {dump_number:05d}: "
                f"min={times.min():.6g}, max={times.max():.6g}, "
                f"n_unique={unique.size}"
            )

    # ------------------------------------------------------------------ main

    def gather_one(self, dump_index: int) -> Path:
        """
        Gather all per-batch scatter contributions for one dump and write the
        final dump_XXXXX.h5.

        For each scatter file only the target dump's HDF5 group is accessed —
        O(n_particles_per_batch × n_fields) bytes per file, not O(file_size).
        The 2-D trace and abundance blocks written by scatter_batches.py are
        sliced into individual named arrays here before writing the final file.
        """
        scatter_files = self.list_scatter_files()
        dump_numbers, chemistry_types, trace_fields, species_names = (
            self._first_file_metadata(scatter_files)
        )

        if dump_index < 0 or dump_index >= len(dump_numbers):
            raise IndexError(
                f"dump_index {dump_index} out of range "
                f"for {len(dump_numbers)} dumps"
            )

        dump_number = dump_numbers[dump_index]
        dump_key = f"dump_{dump_number:05d}"

        # Load canonical (simulation) times from cache so all particles in this
        # dump share exactly the same time value regardless of floating-point drift.
        canonical_times = DumpTimeMapper.load_or_build_dump_time_seconds(
            self.config, dump_numbers
        )
        canonical_seconds = float(canonical_times[dump_index])

        # ---- accumulators ---------------------------------------------------
        ids_list: list[np.ndarray] = []
        # trace_blocks accumulates 2-D slices [n_particles_i, n_trace_fields]
        trace_blocks: list[np.ndarray] = []
        # chem_blocks[ctype] accumulates 2-D slices [n_particles_i, n_species]
        chem_blocks: dict[str, list[np.ndarray]] = {
            ctype: [] for ctype in chemistry_types
        }

        for scatter_path in scatter_files:
            with h5py.File(scatter_path, "r") as src:
                # Fast skip: read a small root attribute before touching groups.
                dumps_present_raw = src.attrs.get("dumps_present_json", "[]")
                if dump_number not in set(json.loads(dumps_present_raw)):
                    continue

                if dump_key not in src:
                    # Defensive fallback if attribute is out of sync.
                    continue

                grp = src[dump_key]
                ids_list.append(grp["id"][:].astype(np.int64))

                # trace block: [n_particles_i, n_trace_fields]
                trace_blocks.append(grp["trace"][:].astype(np.float64))

                # chemistry blocks: [n_particles_i, n_species] per ctype
                for ctype in chemistry_types:
                    if ctype not in grp:
                        continue
                    chem_blocks[ctype].append(
                        grp[ctype]["abundances"][:].astype(np.float64)
                    )

        if not ids_list:
            raise RuntimeError(
                f"No particles found for dump {dump_number:05d} "
                f"across {len(scatter_files)} scatter files."
            )

        # ---- concatenate ----------------------------------------------------
        merged_ids = np.concatenate(ids_list)
        n_total = merged_ids.shape[0]

        # Unpack the stacked trace block into a dict keyed by field name.
        # merged_trace_block shape: [n_total, n_trace_fields]
        merged_trace_block = np.concatenate(trace_blocks, axis=0)
        merged_trace: dict[str, np.ndarray] = {
            name: merged_trace_block[:, i] for i, name in enumerate(trace_fields)
        }
        # Replace raw per-particle times with the single canonical value.
        if "time" in merged_trace:
            merged_trace["time"] = np.full(n_total, canonical_seconds, dtype=np.float64)
            self._assert_single_dump_time(dump_number, merged_trace["time"])

        # Unpack chemistry blocks into {ctype: {species: array}} dicts.
        merged_chem: dict[str, dict[str, np.ndarray]] = {}
        for ctype in chemistry_types:
            blocks = chem_blocks[ctype]
            if blocks:
                merged_block = np.concatenate(blocks, axis=0)
            else:
                n_species = len(species_names.get(ctype, []))
                merged_block = np.empty((0, n_species), dtype=np.float64)

            merged_chem[ctype] = {
                name: merged_block[:, i]
                for i, name in enumerate(species_names[ctype])
            }

        # ---- write ----------------------------------------------------------
        output_path = self.final_output_dir / f"dump_{dump_number:05d}.h5"
        self._write_dump_file(
            output_path,
            dump_number,
            merged_ids,
            merged_trace,
            merged_chem,
            chemistry_types,
        )
        return output_path

    def _write_dump_file(
        self,
        output_path: Path,
        dump_number: int,
        ids: np.ndarray,
        trace_data: dict[str, np.ndarray],
        chemistry_data: dict[str, dict[str, np.ndarray]],
        chemistry_types: list[str],
    ) -> None:
        """Write one merged dump file.  Layout is identical to merge_batches output."""
        kw = self._write_kwargs(self.compression, self.compression_level)

        with h5py.File(output_path, "w") as dst:
            dst.attrs["dump_number"] = int(dump_number)
            dst.attrs["n_particles"] = int(ids.shape[0])
            dst.attrs["chemistry_types_json"] = json.dumps(chemistry_types)

            particles = dst.create_group("particles")
            particles.create_dataset("id", data=ids, dtype=np.int64)

            for name, values in trace_data.items():
                particles.create_dataset(
                    name,
                    data=values.astype(np.float64),
                    dtype=np.float64,
                    **kw,
                )

            if len(chemistry_types) == 1:
                ctype = chemistry_types[0]
                for name, values in chemistry_data[ctype].items():
                    particles.create_dataset(
                        name,
                        data=values.astype(np.float64),
                        dtype=np.float64,
                        **kw,
                    )
            else:
                chem_grp = dst.create_group("chemistry")
                for ctype in chemistry_types:
                    cp = chem_grp.create_group(ctype).create_group("particles")
                    for name, values in chemistry_data[ctype].items():
                        cp.create_dataset(
                            name,
                            data=values.astype(np.float64),
                            dtype=np.float64,
                            **kw,
                        )


class GatherCLI:
    """CLI front-end for ScatterGatherer."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description=(
                "Gather scatter_*.h5 contributions for one dump into dump_XXXXX.h5. "
                "Pass SLURM_ARRAY_TASK_ID as --dump-index."
            )
        )
        p.add_argument("--config", required=True)
        p.add_argument(
            "--dump-index",
            type=int,
            required=True,
            help="0-based dump index to gather (SLURM_ARRAY_TASK_ID).",
        )
        return p.parse_args()

    @classmethod
    def run(cls) -> None:
        args = cls.parse_args()
        config = PipelineConfigManager.load(args.config)
        gatherer = ScatterGatherer(config)
        output = gatherer.gather_one(args.dump_index)
        print(f"Gathered dump file: {output}")


def main() -> None:
    GatherCLI.run()


if __name__ == "__main__":
    main()