from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sarracen
import yaml

from common import BatchPlanner, DumpSelection, PipelineConfigManager


class ParticleIdDiscoveryApp:
    """Parallel particle-ID discovery utility with list/worker/merge modes."""

    def __init__(self, config_path: Path) -> None:
        """Load pipeline configuration once for selected mode execution."""
        self.config_path = config_path.resolve()
        self.config = PipelineConfigManager.load(self.config_path)

    @staticmethod
    def _read_dump_ids(dump_path: Path) -> np.ndarray:
        """Read one PHANTOM dump and return unique int64 particle IDs."""
        sdf, _ = sarracen.read_phantom(str(dump_path))
        ids = np.unique(sdf["iorig"].to_numpy(dtype=np.int64))
        return ids.astype(np.int64, copy=False)

    def run_worker(self, dump_number: int, out_dir: Path) -> None:
        """Worker mode: process one dump number and save per-dump IDs."""
        data_dir = Path(self.config["paths"]["phantom_dump_dir"])
        prefix = self.config["simulation"]["prefix"]
        dump_path = data_dir / f"{prefix}_{dump_number:05d}"
        if not dump_path.is_file():
            raise FileNotFoundError(f"Dump not found: {dump_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"ids_{dump_number:05d}.npy"
        ids = self._read_dump_ids(dump_path)
        np.save(out_file, ids, allow_pickle=False)
        print(f"worker dump={dump_number:05d} ids={len(ids)} -> {out_file}")

    def run_merge(self, dump_list: Path, out_dir: Path, update_config: bool) -> None:
        """Merge mode: combine per-dump worker outputs into one canonical cache."""
        processing = self.config["processing"]
        n_boundary = int(processing.get("n_boundary", 0))
        batch_size = processing.get("batch_size")
        n_batches = processing.get("n_batches")

        dump_numbers = [int(line.strip()) for line in dump_list.read_text(encoding="ascii").splitlines() if line.strip()]
        if not dump_numbers:
            raise RuntimeError(f"Empty dump list: {dump_list}")

        all_ids: set[int] = set()
        for index, dump_number in enumerate(dump_numbers, start=1):
            part_file = out_dir / f"ids_{dump_number:05d}.npy"
            if not part_file.is_file():
                raise FileNotFoundError(f"Missing worker output: {part_file}")
            ids = np.load(part_file, allow_pickle=False)
            all_ids.update(ids.astype(np.int64, copy=False).tolist())
            if index % 50 == 0 or index == len(dump_numbers):
                print(f"merge progress: {index}/{len(dump_numbers)} dumps")

        all_sorted = np.asarray(sorted(all_ids), dtype=np.int64)
        if all_sorted.size == 0:
            raise RuntimeError("Particle discovery produced zero IDs.")

        if n_boundary > 0 and n_boundary >= len(all_sorted):
            raise ValueError(f"processing.n_boundary={n_boundary} removes all {len(all_sorted)} discovered particles")

        particle_ids_file = out_dir / "particle_ids_all.npy"
        np.save(particle_ids_file, all_sorted, allow_pickle=False)

        selected_count = int(len(all_sorted) - n_boundary)
        layout = BatchPlanner.compute_layout(selected_count, batch_size, n_batches)

        metadata = {
            "dump_count": len(dump_numbers),
            "particle_count_all": int(len(all_sorted)),
            "n_boundary": int(n_boundary),
            "particle_count_selected": selected_count,
            "batch_count": int(len(layout)),
            "particle_ids_cache": str(particle_ids_file.resolve()),
            "cache_contains_all_ids": True,
        }

        metadata_file = out_dir / "particle_discovery_metadata.yaml"
        with open(metadata_file, "w", encoding="ascii") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False)

        print("Discovery complete:")
        print(f"  all particles      : {metadata['particle_count_all']}")
        print(f"  after n_boundary   : {metadata['particle_count_selected']}")
        print(f"  computed batches   : {metadata['batch_count']}")
        print(f"  cache file         : {particle_ids_file}")
        print(f"  metadata file      : {metadata_file}")

        if update_config:
            with open(self.config_path, "r", encoding="ascii") as handle:
                config_data: dict = yaml.safe_load(handle)

            if "paths" not in config_data:
                config_data["paths"] = {}
            if "processing" not in config_data:
                config_data["processing"] = {}

            config_data["paths"]["particle_ids_cache"] = str(particle_ids_file.resolve())
            config_data["paths"]["particle_ids_cache_mode"] = "all_ids"
            config_data["processing"]["discovered_particle_count_all"] = int(len(all_sorted))
            config_data["processing"]["discovered_particle_count"] = selected_count
            config_data["processing"]["discovered_batch_count"] = int(len(layout))

            with open(self.config_path, "w", encoding="ascii") as handle:
                yaml.safe_dump(config_data, handle, sort_keys=False)

            print(f"Updated config with discovery fields: {self.config_path}")

    def run_list(self, dump_list: Path) -> None:
        """List mode: write selected dump numbers to plain text for SLURM arrays."""
        selected = DumpSelection.selected_dump_numbers(self.config)
        dump_list.parent.mkdir(parents=True, exist_ok=True)
        dump_list.write_text("\n".join(str(number) for number in selected) + "\n", encoding="ascii")
        print(f"Wrote dump list: {dump_list} ({len(selected)} dumps)")


class ParticleIdDiscoveryCLI:
    """Command-line front-end for ParticleIdDiscoveryApp."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse discovery mode arguments."""
        parser = argparse.ArgumentParser(description="Parallel particle-ID discovery utilities.")
        parser.add_argument("--mode", choices=["list", "worker", "merge"], required=True)
        parser.add_argument("--config", required=True)
        parser.add_argument("--dump-number", type=int)
        parser.add_argument("--dump-list")
        parser.add_argument("--out-dir")
        parser.add_argument("--update-config", action="store_true")
        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        """Dispatch to selected mode while keeping original argument contract."""
        args = cls.parse_args()
        app = ParticleIdDiscoveryApp(config_path=Path(args.config))

        if args.mode == "list":
            if not args.dump_list:
                raise ValueError("--dump-list is required for --mode list")
            app.run_list(dump_list=Path(args.dump_list).resolve())
            return

        if args.mode == "worker":
            if args.dump_number is None or not args.out_dir:
                raise ValueError("--dump-number and --out-dir are required for --mode worker")
            app.run_worker(dump_number=args.dump_number, out_dir=Path(args.out_dir).resolve())
            return

        if args.mode == "merge":
            if not args.dump_list or not args.out_dir:
                raise ValueError("--dump-list and --out-dir are required for --mode merge")
            app.run_merge(
                dump_list=Path(args.dump_list).resolve(),
                out_dir=Path(args.out_dir).resolve(),
                update_config=args.update_config,
            )
            return

        raise RuntimeError(f"Unsupported mode: {args.mode}")


def main() -> None:
    """CLI entrypoint."""
    ParticleIdDiscoveryCLI.run()


if __name__ == "__main__":
    main()
