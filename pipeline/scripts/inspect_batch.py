#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from common import load_pipeline_config


def resolve_batch_dir(config_path: str, batch_dir_arg: str | None) -> Path:
    if batch_dir_arg:
        return Path(batch_dir_arg).expanduser().resolve()

    config = load_pipeline_config(config_path)
    batch_output_dir = config.get("paths", {}).get("batch_output_dir")
    if not batch_output_dir:
        raise KeyError("paths.batch_output_dir is missing from config")
    return Path(batch_output_dir).expanduser().resolve()


def list_batch_files(batch_dir: Path) -> list[Path]:
    files = sorted(batch_dir.glob("batch_*.h5"))
    if not files:
        raise FileNotFoundError(f"No batch_*.h5 files found in {batch_dir}")
    return files


def resolve_batch_file(batch_arg: str | None, batch_files: list[Path]) -> Path:
    if not batch_arg:
        return choose_batch_interactively(batch_files)

    selector = batch_arg.strip()

    explicit_path = Path(selector).expanduser()
    if explicit_path.suffix == ".h5" and explicit_path.exists():
        return explicit_path.resolve()

    if selector.isdigit():
        target_name = f"batch_{int(selector):05d}.h5"
        matches = [path for path in batch_files if path.name == target_name]
        if matches:
            return matches[0]

    if selector.startswith("batch_"):
        target_name = selector if selector.endswith(".h5") else f"{selector}.h5"
        matches = [path for path in batch_files if path.name == target_name]
        if matches:
            return matches[0]

    candidates = [path for path in batch_files if selector in path.name]
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print("Multiple matches found:", file=sys.stderr)
        for idx, path in enumerate(candidates, start=1):
            print(f"  {idx:3d}. {path.name}", file=sys.stderr)
        return choose_batch_interactively(candidates)

    raise FileNotFoundError(f"Could not resolve batch selector '{batch_arg}'")


def choose_batch_interactively(batch_files: list[Path]) -> Path:
    print("Available batches:")
    if len(batch_files) <= 30:
        for idx, path in enumerate(batch_files, start=1):
            print(f"  {idx:3d}. {path.name}")
    else:
        head = batch_files[:10]
        tail = batch_files[-10:]
        for idx, path in enumerate(head, start=1):
            print(f"  {idx:3d}. {path.name}")
        print("  ...")
        offset = len(batch_files) - 10
        for i, path in enumerate(tail, start=1):
            print(f"  {offset + i:3d}. {path.name}")

    while True:
        raw = input(f"Select batch index [1-{len(batch_files)}] or 'q': ").strip().lower()
        if raw == "q":
            raise SystemExit(0)
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(batch_files):
                return batch_files[index - 1]
        print("Invalid selection. Try again.")


def collect_datasets(handle: h5py.File) -> list[tuple[str, h5py.Dataset]]:
    rows: list[tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append((name, obj))

    handle.visititems(visitor)
    rows.sort(key=lambda item: item[0])
    return rows


def format_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def print_summary(rows: list[tuple[str, h5py.Dataset]]) -> None:
    print("\nDataset Summary")
    print("=" * 80)
    print(f"Total datasets: {len(rows)}")
    print(f"{'#':>4}  {'Path':<55} {'Shape':<12} {'DType':<12}")
    print("-" * 80)

    for idx, (name, dataset) in enumerate(rows, start=1):
        shape = format_shape(dataset.shape)
        dtype = str(dataset.dtype)
        short_name = name if len(name) <= 55 else f"...{name[-52:]}"
        print(f"{idx:4d}  {short_name:<55} {shape:<12} {dtype:<12}")


def preview_array(arr: np.ndarray, preview: int) -> str:
    if arr.size == 0:
        return "[]"
    flat = np.ravel(arr)
    shown = flat[:preview]
    values = np.array2string(shown, precision=6, separator=", ", threshold=preview)
    if flat.size > preview:
        return f"{values} ... (total {flat.size} values)"
    return values


def safe_stats(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"

    if np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size == 0:
            return "all values are NaN/inf"
        return f"min={np.min(finite):.6g}, max={np.max(finite):.6g}, mean={np.mean(finite):.6g}"

    if arr.dtype.kind in {"S", "U", "O"}:
        uniques = np.unique(arr)
        return f"unique={len(uniques)}"

    return "stats unavailable"


def interactive_dataset_view(rows: list[tuple[str, h5py.Dataset]], preview: int) -> None:
    print("\nInspect datasets")
    print("Type a dataset index to preview it, 'r' to reprint summary, or 'q' to quit.")

    while True:
        raw = input("dataset> ").strip().lower()
        if raw == "q":
            return
        if raw == "r":
            print_summary(rows)
            continue
        if not raw.isdigit():
            print("Enter an integer index, 'r', or 'q'.")
            continue

        idx = int(raw)
        if not (1 <= idx <= len(rows)):
            print(f"Index out of range (1-{len(rows)}).")
            continue

        name, dataset = rows[idx - 1]
        arr = dataset[()]
        print("-" * 80)
        print(f"Path: /{name}")
        print(f"Shape: {dataset.shape}")
        print(f"DType: {dataset.dtype}")
        print(f"Stats: {safe_stats(np.asarray(arr))}")
        print(f"Preview: {preview_array(np.asarray(arr), preview)}")


def main() -> None:
    # Edit these variables directly before running the script.
    config_path = str(Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml")
    batch_dir_override: str | None = None
    batch_selector: str | None = None
    preview = 8

    batch_dir = resolve_batch_dir(config_path, batch_dir_override)
    batch_files = list_batch_files(batch_dir)
    batch_file = resolve_batch_file(batch_selector, batch_files)

    print(f"Batch directory: {batch_dir}")
    print(f"Selected batch: {batch_file.name}")

    with h5py.File(batch_file, "r") as handle:
        rows = collect_datasets(handle)
        if not rows:
            print("This file contains no datasets.")
            return
        print_summary(rows)
        interactive_dataset_view(rows, preview=preview)


if __name__ == "__main__":
    main()