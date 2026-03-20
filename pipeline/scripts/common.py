from __future__ import annotations

import os
import re
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import sarracen
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CHEMISTRY_ROOT = Path(__file__).resolve().parents[2]

if str(CHEMISTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(CHEMISTRY_ROOT))

from molecules import (
    atoms,
    atoms_plus,
    daughters_2,
    daughters_3,
    daughters_Crich,
    daughters_Orich,
    parents,
)

TRACE_COLUMN_MAP = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "R": "r",
    "DENSITY": "density",
    "TEMP": "temp",
    "AV": "av",
    "TIME": "time",
}

class DumpTimeMapper:
    """Build and use dump-number <-> real-time mappings with fixed-precision keys."""

    SECONDS_PER_YEAR = 31557600.0

    @staticmethod
    def _time_key(seconds: float, decimals: int) -> int:
        """Convert seconds to an integer key using fixed decimal precision."""
        scale = 10**int(decimals)
        return int(np.rint(float(seconds) * scale))

    @staticmethod
    def time_key_decimals(config: dict) -> int:
        """Return configured decimal precision used for time-key matching."""
        processing = config.get("processing", {})
        decimals = int(processing.get("time_key_decimals", 0))
        if decimals < 0:
            raise ValueError("processing.time_key_decimals must be >= 0")
        return decimals

    @staticmethod
    def cache_path(config: dict) -> Path:
        """Return JSON cache path for dump-time mapping."""
        paths = config.get("paths", {})
        explicit = paths.get("dump_time_map_cache")
        if explicit:
            return Path(explicit)
        scratch_root = Path(paths["scratch_root"]).resolve()
        return scratch_root / "dump_time_map_cache.json"

    @classmethod
    def _build_dump_time_seconds(cls, config: dict, dump_numbers: list[int]) -> list[float]:
        """Read PHANTOM dump params and return real-time seconds in dump order.

        Args:
            config (dict): Pipeline configuration.
            dump_numbers (list[int]): List of dump numbers.

        Returns:
            list[float]: List of real-time seconds in dump order.
        """
        paths = config["paths"]
        simulation = config["simulation"]
        data_dir = Path(paths["phantom_dump_dir"])
        prefix = simulation["prefix"]

        values: list[float] = []
        for dump_number in dump_numbers:
            dump_path = data_dir / f"{prefix}_{dump_number:05d}"
            sdf, _ = sarracen.read_phantom(str(dump_path))
            params = getattr(sdf, "_params", {})
            if "time" not in params or "utime" not in params:
                raise KeyError(f"Missing time/utime in dump params for {dump_path}")
            values.append(float(params["time"]) * float(params["utime"]))
        return values

    @classmethod
    def load_or_build_dump_time_seconds(cls, config: dict, dump_numbers: list[int]) -> list[float]:
        """Load dump-time seconds from cache when possible, else build and cache."""
        cache_path = cls.cache_path(config).expanduser().resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="ascii"))
                cached_dumps = [int(value) for value in payload.get("dump_numbers", [])]
                cached_times = [float(value) for value in payload.get("time_seconds", [])]
                if cached_dumps == [int(value) for value in dump_numbers] and len(cached_times) == len(dump_numbers):
                    return cached_times

                # Common case: cache covers the full selected dump list, while caller requests
                # a subset. Reuse by keyed lookup.
                cache_lookup = {int(dump): float(seconds) for dump, seconds in zip(cached_dumps, cached_times)}
                if all(int(dump) in cache_lookup for dump in dump_numbers):
                    return [cache_lookup[int(dump)] for dump in dump_numbers]
            except Exception:
                # Corrupt/partial cache is rebuilt below.
                pass

        time_seconds = cls._build_dump_time_seconds(config, dump_numbers)
        payload = {
            "dump_numbers": [int(value) for value in dump_numbers],
            "time_seconds": [float(value) for value in time_seconds],
        }
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload), encoding="ascii")
        temp_path.replace(cache_path)
        return time_seconds

    @classmethod
    def dump_lookup_from_seconds(
        cls,
        dump_numbers: list[int],
        dump_time_seconds: list[float],
        decimals: int,
    ) -> dict[int, int]:
        """Build key->dump lookup and fail if duplicate keys appear.

        Args:
            dump_numbers (list[int]): List of dump numbers.
            dump_time_seconds (list[float]): List of real-time seconds in dump order.
            decimals (int): Decimal precision for time-key matching.

        Returns:
            dict[int, int]: Dictionary mapping time keys to dump numbers.
        """
        lookup: dict[int, int] = {}
        for dump_number, seconds in zip(dump_numbers, dump_time_seconds):
            key = cls._time_key(seconds, decimals)
            if key in lookup and lookup[key] != int(dump_number):
                raise ValueError(
                    f"Time-key collision at precision {decimals} for dumps "
                    f"{lookup[key]:05d} and {int(dump_number):05d}"
                )
            lookup[key] = int(dump_number)
        return lookup

    @classmethod
    def map_time_values_to_dump_numbers(
        cls,
        time_values: np.ndarray,
        dump_numbers: list[int],
        dump_time_seconds: list[float],
        decimals: int,
        unit_hint: str | None = None,
    ) -> np.ndarray:
        """Map time values (years or seconds) to dump numbers using fixed-precision keys.

        Two hypotheses are tested:
        - values are already in seconds
        - values are in years and need conversion to seconds
        The hypothesis with more exact key matches is selected.

        Args:
            time_values (np.ndarray): Array of time values.
            dump_numbers (list[int]): List of dump numbers.
            dump_time_seconds (list[float]): List of real-time seconds in dump order.
            decimals (int): Decimal precision for time-key matching.
            unit_hint (str | None, optional): Hint for the unit of the time values. Defaults to None.

        Returns:
            np.ndarray: Array of dump numbers corresponding to the time values.
        """
        times = np.asarray(time_values, dtype=np.float64)
        unit_mode = "auto" if unit_hint is None else str(unit_hint).strip().lower()
        if unit_mode not in {"auto", "seconds", "years"}:
            raise ValueError(f"Unsupported unit_hint={unit_hint!r}; expected 'auto', 'seconds', or 'years'")

        lookup = cls.dump_lookup_from_seconds(dump_numbers, dump_time_seconds, decimals)
        dump_seconds = np.asarray(dump_time_seconds, dtype=np.float64)
        dump_numbers_array = np.asarray(dump_numbers, dtype=np.int32)

        keys_seconds = np.asarray([cls._time_key(value, decimals) for value in times], dtype=np.int64)
        keys_years = np.asarray([cls._time_key(value * cls.SECONDS_PER_YEAR, decimals) for value in times], dtype=np.int64)

        matches_seconds = int(sum(1 for key in keys_seconds if int(key) in lookup))
        matches_years = int(sum(1 for key in keys_years if int(key) in lookup))

        if unit_mode == "seconds":
            chosen_keys = keys_seconds
            chosen_seconds = times
        elif unit_mode == "years":
            chosen_keys = keys_years
            chosen_seconds = times * cls.SECONDS_PER_YEAR
        elif matches_seconds == matches_years:
            if matches_seconds == len(times):
                chosen_keys = keys_seconds
                chosen_seconds = times
            else:
                raise ValueError(
                    "Ambiguous time-unit detection for EV TIME column: "
                    f"seconds_matches={matches_seconds}, years_matches={matches_years}, rows={len(times)}"
                )
        elif matches_seconds > matches_years:
            chosen_keys = keys_seconds
            chosen_seconds = times
        else:
            chosen_keys = keys_years
            chosen_seconds = times * cls.SECONDS_PER_YEAR

        mapped = []
        for index, key in enumerate(chosen_keys.tolist()):
            if key not in lookup:
                nearest_index = int(np.argmin(np.abs(dump_seconds - chosen_seconds[index])))
                raise ValueError(
                    "Could not map EV TIME value to a configured dump using fixed precision: "
                    f"row={index}, time={times[index]:.12g}, key={key}, decimals={decimals}, "
                    f"unit_mode={unit_mode}, nearest_dump={int(dump_numbers_array[nearest_index]):05d}, "
                    f"nearest_time_seconds={dump_seconds[nearest_index]:.12g}, "
                    f"absdiff_seconds={abs(dump_seconds[nearest_index] - chosen_seconds[index]):.12g}"
                )
            mapped.append(lookup[key])
        return np.asarray(mapped, dtype=np.int32)


class PipelineConfigManager:
    """Load and normalise pipeline configuration dictionaries."""

    @staticmethod
    def _expand_value(value):
        """Go through all settings and replace environment variables and home folder shortcuts with their full paths.

        Args:
            value (Any): Value to expand.

        Returns:
            Any: Expanded value.
        """
        if isinstance(value, str):
            return os.path.expandvars(os.path.expanduser(value))
        if isinstance(value, list):
            return [PipelineConfigManager._expand_value(item) for item in value]
        if isinstance(value, dict):
            return {key: PipelineConfigManager._expand_value(item) for key, item in value.items()}
        return value

    @staticmethod
    def _resolve_paths_with_base(config: dict) -> dict:
        """Resolve relative entries in paths.* against paths.base_path."""
        paths = config.get("paths")
        if not isinstance(paths, dict):
            return config

        base_value = paths.get("base_path")
        if not isinstance(base_value, str) or not base_value.strip():
            return config

        base_path = Path(base_value).expanduser()
        if not base_path.is_absolute():
            base_path = (PIPELINE_ROOT / base_path).resolve()
        else:
            base_path = base_path.resolve()

        paths["base_path"] = str(base_path)

        for key, value in list(paths.items()):
            if key == "base_path" or not isinstance(value, str) or not value.strip():
                continue
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                paths[key] = str((base_path / resolved).resolve())

        return config

    @classmethod
    def load(cls, config_path: str | Path) -> dict:
        """Load YAML config and attach derived root metadata fields."""
        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="ascii") as handle:
            config = yaml.safe_load(handle)
        config = cls._expand_value(config)
        config = cls._resolve_paths_with_base(config)
        config["config_path"] = str(config_path)
        config["pipeline_root"] = str(PIPELINE_ROOT)
        config["chemistry_root"] = str(CHEMISTRY_ROOT)
        return config


class SpeciesCatalog:
    """Species-name helpers used when building chemistry runtime inputs."""

    @staticmethod
    def list_for_chemistry(chemistry_type: str) -> list[str]:
        """Return the ordered species list expected by evolve_output.pl.
        
        Reads all 668 species from the .specs file and filters out:
        - Species starting with 'G' (grains)
        - Species containing 'Y'
        - A hardcoded list of undefined species

        Args:
            chemistry_type (str): Type of chemistry.

        Returns:
            list[str]: List of species names.
        """
        specs_filename = SpeciesCatalog.species_filename(chemistry_type)
        specs_path = CHEMISTRY_ROOT / "evolving_model" / specs_filename
        
        if not specs_path.is_file():
            raise FileNotFoundError(f"Species catalog not found: {specs_path}")
        
        species = []

        undefined_ab_species = [
        "F+",
        "COOCH3+",
        "C2H4CN",
        "HC2O",
        "HCCN",
        "CH3COOH+",
        "COOCH3",
        "CH3COOH2+",
        "CH3COOH",
        "CH3CO",
        "COOH",
        "He+",
        "HF+"
    ]

        with open(specs_path, "r", encoding="ascii") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 2:
                    continue
                species_name = parts[1]
                # Filter: exclude species starting with 'G' (grains)
                if species_name.startswith("G"):
                    continue
                # Filter: exclude species containing 'Y'
                if "Y" in species_name:
                    continue
                if species_name in undefined_ab_species:
                    continue
                species.append(species_name)
        
        return species

    @staticmethod
    def species_filename(chemistry_type: str) -> str:
        """Return species file name selected by chemistry type."""
        return f"rate12_complex_atomic_{chemistry_type}.specs"

    @staticmethod
    def normalise_name(name: str) -> str:
        """Convert chemistry column names into HDF5-safe dataset names."""
        cleaned = name.strip().lower()
        cleaned = cleaned.replace("+", "_plus")
        cleaned = cleaned.replace("-", "_minus")
        cleaned = cleaned.replace("/", "_")
        cleaned = cleaned.replace("(", "_")
        cleaned = cleaned.replace(")", "_")
        cleaned = cleaned.replace(".", "_")
        cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned


class DumpSelection:
    """Dump listing and selection logic based on pipeline processing settings."""

    @staticmethod
    def list_dump_numbers(data_dir: str | Path, prefix: str) -> list[int]:
        """Scan data directory and return sorted dump numbers for prefix_00000 files."""
        data_dir = Path(data_dir)
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{5}})$")
        dump_numbers = []
        for entry in data_dir.iterdir():
            match = pattern.match(entry.name)
            if match and entry.is_file():
                dump_numbers.append(int(match.group(1)))
        return sorted(dump_numbers)

    @classmethod
    def selected_dump_numbers(cls, config: dict) -> list[int]:
        """Apply start/end/step filters and return configured dump sequence."""
        paths = config["paths"]
        processing = config["processing"]
        prefix = config["simulation"]["prefix"]
        data_dir = Path(paths["phantom_dump_dir"])

        step = int(processing.get("dump_step", 1))
        if step <= 0:
            raise ValueError("processing.dump_step must be > 0")

        all_dump_numbers = cls.list_dump_numbers(data_dir, prefix)
        if not all_dump_numbers:
            raise FileNotFoundError(f"No dumps found in {data_dir} for prefix {prefix}")

        start_dump = int(processing.get("start_dump", all_dump_numbers[0]))
        end_dump = int(processing.get("end_dump", all_dump_numbers[-1]))
        selected = [number for number in all_dump_numbers if start_dump <= number <= end_dump]
        selected = selected[::step]
        if len(selected) < 2:
            raise FileNotFoundError(
                f"Need at least 2 dumps in configured range [{start_dump}, {end_dump}] for prefix {prefix}"
            )
        return selected

    @classmethod
    def target_dump_numbers(cls, config: dict) -> list[int]:
        """Return target dumps for chemistry batching and merged output files."""
        return cls.selected_dump_numbers(config)

class ParticleIdStore:
    """Discover, cache, and validate selected particle ID arrays."""

    @staticmethod
    def cache_path(config: dict) -> Path | None:
        """Return configured particle ID cache path if configured."""
        paths = config.get("paths", {})
        cache = paths.get("particle_ids_cache")
        if not cache:
            return None
        return Path(cache)

    @staticmethod
    def apply_n_boundary(particle_ids: np.ndarray, n_boundary: int) -> np.ndarray:
        """Drop first n_boundary particle IDs using configured sorted ordering."""
        if n_boundary <= 0:
            return particle_ids
        if n_boundary >= len(particle_ids):
            raise ValueError(f"processing.n_boundary={n_boundary} removes all {len(particle_ids)} particles")
        return particle_ids[n_boundary:]

    @classmethod
    def load_cached(cls, config: dict) -> np.ndarray | None:
        """Load particle IDs from cache, rejecting stale caches with wrong expected size."""
        cache_path = cls.cache_path(config)
        if cache_path is None or not cache_path.is_file():
            return None
        particle_ids = np.load(cache_path, allow_pickle=False)
        if particle_ids.dtype != np.int64:
            particle_ids = particle_ids.astype(np.int64)
        expected_all = config.get("processing", {}).get("discovered_particle_count_all")
        if expected_all is not None and int(expected_all) != int(len(particle_ids)):
            # Old cache format stored n_boundary-filtered IDs; treat as stale to avoid wrong selection.
            return None
        return particle_ids

    @classmethod
    def discover(cls, config: dict, show_progress: bool = False) -> np.ndarray:
        """Discover unique particle IDs by scanning all selected PHANTOM dumps."""
        cached = cls.load_cached(config)
        if cached is not None:
            n_boundary = int(config["processing"].get("n_boundary", 0))
            return cls.apply_n_boundary(cached, n_boundary)

        simulation = config["simulation"]
        paths = config["paths"]
        prefix = simulation["prefix"]
        data_dir = Path(paths["phantom_dump_dir"])

        dump_numbers = DumpSelection.selected_dump_numbers(config)

        unique_ids: set[int] = set()
        for dump_number in dump_numbers:
            dump_path = data_dir / f"{prefix}_{dump_number:05d}"
            sdf, _ = sarracen.read_phantom(str(dump_path))
            unique_ids.update(np.unique(sdf["iorig"].to_numpy(dtype=np.int64)).tolist())
            if show_progress:
                print(f"Scanned dump {dump_number:05d}", file=sys.stderr, flush=True)

        particle_ids = np.asarray(sorted(unique_ids), dtype=np.int64)
        if particle_ids.size == 0:
            raise RuntimeError("Particle ID discovery found zero particles across all configured dumps.")

        n_boundary = int(config["processing"].get("n_boundary", 0))
        return cls.apply_n_boundary(particle_ids, n_boundary)


class BatchPlanner:
    """Compute batch partitioning and provide per-batch particle selections."""

    @staticmethod
    def compute_layout(total_particles: int, batch_size: int | None, n_batches: int | None) -> list[tuple[int, int]]:
        """Build [start, end) slices for batch partitioning from configured strategy."""
        if batch_size is None and n_batches is None:
            raise ValueError("At least one of batch_size or n_batches must be configured.")

        if batch_size is not None and batch_size <= 0:
            raise ValueError("processing.batch_size must be > 0")
        if n_batches is not None and n_batches <= 0:
            raise ValueError("processing.n_batches must be > 0")

        layout: list[tuple[int, int]] = []

        if batch_size is not None:
            max_particles = total_particles if n_batches is None else min(total_particles, batch_size * n_batches)
            for start in range(0, max_particles, batch_size):
                end = min(start + batch_size, max_particles)
                layout.append((start, end))
            return layout

        base = total_particles // n_batches
        remainder = total_particles % n_batches
        start = 0
        for batch_index in range(n_batches):
            width = base + (1 if batch_index < remainder else 0)
            end = start + width
            if start < end:
                layout.append((start, end))
            start = end
        return layout

    @classmethod
    def batch_particle_ids(cls, config: dict, batch_index: int) -> tuple[np.ndarray, dict]:
        """Return IDs and metadata for one batch index using cached discovery data."""
        particle_ids_all = ParticleIdStore.load_cached(config)
        if particle_ids_all is None:
            raise RuntimeError(
                "Particle ID cache is missing or stale for this config. "
                "Run ./workflow/submit_discover_particle_ids.sh to rebuild cache."
            )
        n_boundary = int(config["processing"].get("n_boundary", 0))
        particle_ids = ParticleIdStore.apply_n_boundary(particle_ids_all, n_boundary)
        processing = config["processing"]
        batch_size = processing.get("batch_size")
        n_batches = processing.get("n_batches")
        layout = cls.compute_layout(len(particle_ids), batch_size, n_batches)
        if batch_index < 0 or batch_index >= len(layout):
            raise IndexError(f"Batch index {batch_index} is out of range for {len(layout)} batch(es)")

        start, end = layout[batch_index]
        selected = particle_ids[start:end]
        metadata = {
            "batch_index": batch_index,
            "batch_count": len(layout),
            "selection_start": start,
            "selection_end": end,
            "selected_particles": len(selected),
            "total_particles": len(particle_ids),
        }
        return selected, metadata

    @classmethod
    def batch_count(cls, config: dict, show_progress: bool = False) -> int:
        """Compute total number of batches from cached or discovered particle counts."""
        processing = config.get("processing", {})
        n_boundary = int(processing.get("n_boundary", 0))

        count_all = processing.get("discovered_particle_count_all")
        if count_all is not None and not show_progress:
            total = int(count_all) - n_boundary
            if total <= 0:
                raise ValueError(
                    f"processing.n_boundary={n_boundary} removes all discovered particles ({int(count_all)})"
                )
        else:
            particle_ids_all = ParticleIdStore.load_cached(config)
            if particle_ids_all is None:
                particle_ids = ParticleIdStore.discover(config, show_progress=show_progress)
                total = int(len(particle_ids))
            else:
                total = int(len(ParticleIdStore.apply_n_boundary(particle_ids_all, n_boundary)))
        batch_size = processing.get("batch_size")
        n_batches = processing.get("n_batches")
        layout = cls.compute_layout(total, batch_size, n_batches)
        return len(layout)


class FileSystemTools:
    """File system helpers shared by pipeline scripts."""

    @staticmethod
    def ensure_clean_directory(path: str | Path) -> Path:
        """Create a directory if needed and return it as a Path object."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def ensure_link(source: str | Path, destination: str | Path) -> None:
        """Create or replace symlink destination -> source."""
        source = Path(source)
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.symlink(source, destination)
