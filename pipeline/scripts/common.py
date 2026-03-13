from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import sarracen
import yaml


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CHEMISTRY_ROOT = Path(__file__).resolve().parents[2]

if str(CHEMISTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(CHEMISTRY_ROOT))

from molecules import (  # noqa: E402
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


def _expand_value(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    return value


def _resolve_paths_with_base(config: dict) -> dict:
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


def load_pipeline_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="ascii") as handle:
        config = yaml.safe_load(handle)
    config = _expand_value(config)
    config = _resolve_paths_with_base(config)
    config["config_path"] = str(config_path)
    config["pipeline_root"] = str(PIPELINE_ROOT)
    config["chemistry_root"] = str(CHEMISTRY_ROOT)
    return config


def get_species_for_chemistry(chemistry_type: str) -> list[str]:
    daughters = daughters_Orich if chemistry_type == "Orich" else daughters_Crich
    species = parents + daughters + daughters_2 + daughters_3 + atoms + atoms_plus
    return [str(name) for name in species]


def get_species_filename(chemistry_type: str) -> str:
    return f"rate12_complex_atomic_{chemistry_type}.specs"


def normalise_species_name(name: str) -> str:
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


def list_dump_numbers(data_dir: str | Path, prefix: str) -> list[int]:
    data_dir = Path(data_dir)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{5}})$")
    dump_numbers = []
    for entry in data_dir.iterdir():
        match = pattern.match(entry.name)
        if match and entry.is_file():
            dump_numbers.append(int(match.group(1)))
    return sorted(dump_numbers)


def get_selected_dump_numbers(config: dict) -> list[int]:
    paths = config["paths"]
    processing = config["processing"]
    prefix = config["simulation"]["prefix"]
    data_dir = Path(paths["phantom_dump_dir"])

    step = int(processing.get("dump_step", 1))
    if step <= 0:
        raise ValueError("processing.dump_step must be > 0")

    all_dump_numbers = list_dump_numbers(data_dir, prefix)
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


def get_target_dumps(config: dict) -> list[int]:
    selected = get_selected_dump_numbers(config)
    return selected[1:]


def get_particle_ids_cache_path(config: dict) -> Path | None:
    paths = config.get("paths", {})
    cache = paths.get("particle_ids_cache")
    if not cache:
        return None
    return Path(cache)


def _apply_n_boundary(particle_ids: np.ndarray, n_boundary: int) -> np.ndarray:
    if n_boundary <= 0:
        return particle_ids
    if n_boundary >= len(particle_ids):
        raise ValueError(f"processing.n_boundary={n_boundary} removes all {len(particle_ids)} particles")
    return particle_ids[n_boundary:]


def load_cached_particle_ids(config: dict) -> np.ndarray | None:
    cache_path = get_particle_ids_cache_path(config)
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


def discover_particle_ids(config: dict, show_progress: bool = False) -> np.ndarray:
    cached = load_cached_particle_ids(config)
    if cached is not None:
        n_boundary = int(config["processing"].get("n_boundary", 0))
        return _apply_n_boundary(cached, n_boundary)

    simulation = config["simulation"]
    paths = config["paths"]
    prefix = simulation["prefix"]
    data_dir = Path(paths["phantom_dump_dir"])

    dump_numbers = get_selected_dump_numbers(config)

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
    return _apply_n_boundary(particle_ids, n_boundary)


def compute_batch_layout(total_particles: int, batch_size: int | None, n_batches: int | None) -> list[tuple[int, int]]:
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


def get_batch_particle_ids(config: dict, batch_index: int) -> tuple[np.ndarray, dict]:
    particle_ids_all = load_cached_particle_ids(config)
    if particle_ids_all is None:
        raise RuntimeError(
            "Particle ID cache is missing or stale for this config. "
            "Run ./submit_discover_particle_ids.sh to rebuild cache."
        )
    n_boundary = int(config["processing"].get("n_boundary", 0))
    particle_ids = _apply_n_boundary(particle_ids_all, n_boundary)
    processing = config["processing"]
    batch_size = processing.get("batch_size")
    n_batches = processing.get("n_batches")
    layout = compute_batch_layout(len(particle_ids), batch_size, n_batches)
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


def get_batch_count(config: dict, show_progress: bool = False) -> int:
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
        particle_ids_all = load_cached_particle_ids(config)
        if particle_ids_all is None:
            particle_ids = discover_particle_ids(config, show_progress=show_progress)
            total = int(len(particle_ids))
        else:
            total = int(len(_apply_n_boundary(particle_ids_all, n_boundary)))
    batch_size = processing.get("batch_size")
    n_batches = processing.get("n_batches")
    layout = compute_batch_layout(total, batch_size, n_batches)
    return len(layout)


def ensure_clean_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
