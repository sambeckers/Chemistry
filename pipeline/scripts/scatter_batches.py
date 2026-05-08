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
        dumps_present_json   "[3, 7, 12, …]'  ← only dumps with ≥1 particle
      /dump_00003/
        attrs: dump_number, n_particles
        id           (int64)
        x, y, z, …  (float64, trace datasets)
        Crich/       (group, one dataset per species)
          CO2, H2O, …

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

            trace_particles = src["trace/particles"]
            trace_names: list[str] = list(trace_particles.keys())
            # shape after slicing: [n_valid, n_dumps]
            trace_arrays: dict[str, np.ndarray] = {
                name: trace_particles[name][:][valid_mask] for name in trace_names
            }

            chem_arrays: dict[str, dict[str, np.ndarray]] = {}
            for ctype in chemistry_types:
                chem_grp = src[f"chemistry/{ctype}/particles"]
                chem_arrays[ctype] = {
                    name: chem_grp[name][:][valid_mask] for name in chem_grp.keys()
                }

        # Use the first trace dataset to detect which particles exist per dump.
        reference_arr = trace_arrays[trace_names[0]]  # [n_valid, n_dumps]

        kw = self._write_kwargs(self.compression, self.compression_level)
        dumps_present: list[int] = []

        with h5py.File(output_path, "w") as dst:
            dst.attrs["batch_index"] = batch_index
            dst.attrs["dump_numbers_json"] = json.dumps(dump_numbers)
            dst.attrs["chemistry_types_json"] = json.dumps(chemistry_types)

            for t_idx, dump_number in enumerate(dump_numbers):
                present = ~np.isnan(reference_arr[:, t_idx])
                if not np.any(present):
                    continue

                dumps_present.append(dump_number)
                ids_here = valid_ids[present]

                grp = dst.create_group(f"dump_{dump_number:05d}")
                grp.attrs["dump_number"] = dump_number
                grp.attrs["n_particles"] = int(ids_here.shape[0])
                grp.create_dataset("id", data=ids_here, dtype=np.int64)

                for name, arr in trace_arrays.items():
                    col = arr[:, t_idx][present]
                    grp.create_dataset(
                        name,
                        data=col.astype(np.float64),
                        dtype=np.float64,
                        **kw,
                    )

                for ctype in chemistry_types:
                    chem_subgrp = grp.create_group(ctype)
                    for name, arr in chem_arrays[ctype].items():
                        col = arr[:, t_idx][present]
                        chem_subgrp.create_dataset(
                            name,
                            data=col.astype(np.float64),
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
                "Scatter one batch_*.h5 into a dump-organised scatter_*.h5. "
                "Pass SLURM_ARRAY_TASK_ID as --batch-index."
            )
        )
        p.add_argument("--config", required=True)
        p.add_argument(
            "--batch-index",
            type=int,
            required=True,
            help="0-based index of the batch file to process (SLURM_ARRAY_TASK_ID).",
        )
        return p.parse_args()

    @classmethod
    def run(cls) -> None:
        args = cls.parse_args()
        config = PipelineConfigManager.load(args.config)
        scatterer = BatchScatterer(config)
        output = scatterer.scatter_one(args.batch_index)
        print(f"Scatter file written: {output}")


def main() -> None:
    ScatterCLI.run()


if __name__ == "__main__":
    main()
