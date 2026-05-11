"""
scatter_batches.py — Scatter phase of the two-stage merge.

Each SLURM array task (0 … n_batches-1) reads ONE batch_*.h5 file entirely
into RAM in a single sequential pass, then writes ONE scatter_*.h5 organised
by dump group.  Tasks are fully independent; there is zero write contention.

The scatter file layout is::

    scatter_NNNNN.h5
      attrs:
        batch_index          int
        dump_numbers_json    "[0, 1, 2, …]"
        chemistry_types_json '["Crich"]'
        dumps_present_json   "[3, 7, 12, …]"  ← only dumps with ≥1 particle
        trace_fields_json    '["x", "y", "z", "temp", …]'
        species_json         '{"Crich": ["c", "c2", …]}'
      /dump_00003/
        attrs: dump_number, n_particles
        id            (int64,   1-D: n_particles)
        trace         (float64, 2-D: n_particles × n_trace_fields)
        Crich/
          abundances  (float64, 2-D: n_particles × n_species)

Storing chemistry and trace data as 2-D blocks rather than one dataset per
species/field reduces the number of HDF5 objects per scatter file from
~780 000 down to ~5 000, eliminating the metadata-overhead bloat that
inflated the naive implementation to ~5× the source data volume.

The dumps_present_json attribute lets the gather phase skip scatter files
that have no data for a given dump without reading any group structure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import FileSystemTools, PipelineConfigManager


class BatchScatterer:
    """Transpose one batch_*.h5 from particle-major to dump-major layout."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.batch_output_dir = Path(config["paths"]["batch_output_dir"]).resolve()
        scatter_path = config["paths"].get(
            "scatter_output_dir",
            str(Path(config["paths"]["batch_output_dir"]).parent / "scatter"),
        )
        self.scatter_dir = FileSystemTools.ensure_clean_directory(scatter_path)
        self.compression = config["processing"].get("compression", "gzip")
        self.compression_level = config["processing"].get("compression_level", 4)

    # ------------------------------------------------------------------ helpers

    def list_batch_files(self) -> list[Path]:
        files = sorted(self.batch_output_dir.glob("batch_*.h5"))
        if not files:
            raise FileNotFoundError(f"No batch files in {self.batch_output_dir}")
        return files

    @staticmethod
    def _write_kwargs(compression: str | None, level: int | None) -> dict:
        kw: dict = {}
        if compression:
            kw["compression"] = compression
            if compression == "gzip" and level is not None:
                kw["compression_opts"] = int(level)
        return kw

    # ------------------------------------------------------------------ main

    def scatter_one(self, batch_index: int) -> Path:
        """
        Transpose one batch file into a dump-organised scatter file.

        The entire batch file is read into RAM in a single sequential pass so
        that Lustre sees one large contiguous read instead of thousands of
        column seeks.  All per-dump slices are then written out to a new file
        that the gather phase can read with targeted group lookups.

        Chemistry and trace data are stored as 2-D blocks (n_particles ×
        n_fields) rather than one dataset per field.  This keeps the HDF5
        object count low and preserves gzip effectiveness on larger arrays.
        """
        batch_files = self.list_batch_files()
        if batch_index < 0 or batch_index >= len(batch_files):
            raise IndexError(
                f"batch_index {batch_index} out of range "
                f"for {len(batch_files)} batch files"
            )

        batch_path = batch_files[batch_index]
        output_path = self.scatter_dir / f"scatter_{batch_index:05d}.h5"

        # ---- load the entire batch into RAM (one sequential Lustre read) -----
        with h5py.File(batch_path, "r") as src:
            dump_numbers: list[int] = [
                int(v) for v in json.loads(src.attrs["dump_numbers_json"])
            ]
            chemistry_types: list[str] = list(
                json.loads(src.attrs["chemistry_types_json"])
            )

            particle_ids: np.ndarray = src["particles/id"][:]
            valid_mask = particle_ids > 0
            valid_ids = particle_ids[valid_mask].astype(np.int64)

            # ---- trace: read all fields, record ordered name list -----------
            trace_particles = src["trace/particles"]
            trace_names: list[str] = sorted(trace_particles.keys())
            # trace_block shape: [n_valid, n_trace_fields, n_dumps]
            # Build as [n_valid, n_dumps] per field first, stack at the end.
            trace_arrays: list[np.ndarray] = [
                trace_particles[name][:][valid_mask] for name in trace_names
            ]
            # Stack → [n_valid, n_trace_fields, n_dumps], then transpose to
            # [n_dumps, n_valid, n_trace_fields] for cheap column slicing later.
            trace_block = np.stack(trace_arrays, axis=1)
            # trace_block[particle, field, dump]  →  we want [dump, particle, field]
            trace_block = trace_block.transpose(2, 0, 1)
            # shape: [n_dumps, n_valid, n_trace_fields]

            # Use the first trace field to detect which particles exist per dump.
            # A particle is "present" at a dump when its value is not NaN.
            reference_field_idx = 0  # arbitrary; all trace fields share the mask

            # ---- chemistry: same treatment ----------------------------------
            # chem_blocks[ctype] shape: [n_dumps, n_valid, n_species]
            chem_blocks: dict[str, np.ndarray] = {}
            species_names: dict[str, list[str]] = {}
            for ctype in chemistry_types:
                chem_grp = src[f"chemistry/{ctype}/particles"]
                names = sorted(chem_grp.keys())
                species_names[ctype] = names
                arrays = [chem_grp[name][:][valid_mask] for name in names]
                # stack → [n_valid, n_species, n_dumps], then → [n_dumps, n_valid, n_species]
                chem_blocks[ctype] = np.stack(arrays, axis=1).transpose(2, 0, 1)

        kw = self._write_kwargs(self.compression, self.compression_level)
        dumps_present: list[int] = []

        with h5py.File(output_path, "w") as dst:
            # ---- file-level metadata ----------------------------------------
            dst.attrs["batch_index"] = batch_index
            dst.attrs["dump_numbers_json"] = json.dumps(dump_numbers)
            dst.attrs["chemistry_types_json"] = json.dumps(chemistry_types)
            # Store field/species name lists once at the file level so the
            # gather phase does not need to open every dump group to find them.
            dst.attrs["trace_fields_json"] = json.dumps(trace_names)
            dst.attrs["species_json"] = json.dumps(species_names)

            # ---- per-dump groups --------------------------------------------
            for t_idx, dump_number in enumerate(dump_numbers):
                present = ~np.isnan(trace_block[t_idx, :, reference_field_idx])
                if not np.any(present):
                    continue

                dumps_present.append(dump_number)
                ids_here = valid_ids[present]

                grp = dst.create_group(f"dump_{dump_number:05d}")
                grp.attrs["dump_number"] = dump_number
                grp.attrs["n_particles"] = int(ids_here.shape[0])

                grp.create_dataset("id", data=ids_here, dtype=np.int64)

                # trace block for this dump: [n_present, n_trace_fields]
                trace_slice = trace_block[t_idx][present]
                grp.create_dataset(
                    "trace",
                    data=trace_slice.astype(np.float64),
                    dtype=np.float64,
                    **kw,
                )

                # chemistry blocks
                for ctype in chemistry_types:
                    # [n_present, n_species]
                    chem_slice = chem_blocks[ctype][t_idx][present]
                    chem_subgrp = grp.create_group(ctype)
                    chem_subgrp.create_dataset(
                        "abundances",
                        data=chem_slice.astype(np.float64),
                        dtype=np.float64,
                        **kw,
                    )

            # Written last so it reflects only dumps that actually had particles.
            dst.attrs["dumps_present_json"] = json.dumps(dumps_present)

        return output_path


class ScatterCLI:
    """CLI front-end for BatchScatterer."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description=(
                "Scatter one or more batch_*.h5 files into dump-organised "
                "scatter_*.h5 files.  Pass a single index via --batch-index "
                "or multiple indices via --batch-indices for serial batching "
                "within one SLURM task."
            )
        )
        p.add_argument("--config", required=True)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--batch-index",
            type=int,
            help="Single 0-based batch index (legacy / single-task mode).",
        )
        group.add_argument(
            "--batch-indices",
            type=int,
            nargs="+",
            metavar="IDX",
            help="One or more 0-based batch indices to process serially.",
        )
        return p.parse_args()

    @classmethod
    def run(cls) -> None:
        args = cls.parse_args()
        indices: list[int] = (
            [args.batch_index]
            if args.batch_index is not None
            else args.batch_indices
        )
        config = PipelineConfigManager.load(args.config)
        scatterer = BatchScatterer(config)
        for idx in indices:
            output = scatterer.scatter_one(idx)
            print(f"Scatter file written: {output}")


def main() -> None:
    ScatterCLI.run()


if __name__ == "__main__":
    main()