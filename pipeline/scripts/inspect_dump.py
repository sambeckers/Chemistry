#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from common import load_pipeline_config


def resolve_dump_dir(config_path: str, dump_dir_arg: str | None) -> Path:
    if dump_dir_arg:
        return Path(dump_dir_arg).expanduser().resolve()

    config = load_pipeline_config(config_path)
    final_output_dir = config.get("paths", {}).get("final_output_dir")
    if not final_output_dir:
        raise KeyError("paths.final_output_dir is missing from config")
    return Path(final_output_dir).expanduser().resolve()


def list_dump_files(dump_dir: Path) -> list[Path]:
    files = sorted(dump_dir.glob("dump_*.h5"))
    if not files:
        raise FileNotFoundError(f"No dump_*.h5 files found in {dump_dir}")
    return files


def resolve_dump_file(dump_arg: str | None, dump_files: list[Path]) -> Path:
    if not dump_arg:
        return choose_dump_interactively(dump_files)

    selector = dump_arg.strip()

    explicit_path = Path(selector).expanduser()
    if explicit_path.suffix == ".h5" and explicit_path.exists():
        return explicit_path.resolve()

    if selector.isdigit():
        target_name = f"dump_{int(selector):05d}.h5"
        matches = [path for path in dump_files if path.name == target_name]
        if matches:
            return matches[0]

    if selector.startswith("dump_"):
        target_name = selector if selector.endswith(".h5") else f"{selector}.h5"
        matches = [path for path in dump_files if path.name == target_name]
        if matches:
            return matches[0]

    candidates = [path for path in dump_files if selector in path.name]
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print("Multiple matches found:", file=sys.stderr)
        for idx, path in enumerate(candidates, start=1):
            print(f"  {idx:3d}. {path.name}", file=sys.stderr)
        return choose_dump_interactively(candidates)

    raise FileNotFoundError(f"Could not resolve dump selector '{dump_arg}'")


def choose_dump_interactively(dump_files: list[Path]) -> Path:
    print("Available dumps:")
    if len(dump_files) <= 30:
        for idx, path in enumerate(dump_files, start=1):
            print(f"  {idx:3d}. {path.name}")
    else:
        head = dump_files[:10]
        tail = dump_files[-10:]
        for idx, path in enumerate(head, start=1):
            print(f"  {idx:3d}. {path.name}")
        print("  ...")
        offset = len(dump_files) - 10
        for i, path in enumerate(tail, start=1):
            print(f"  {offset + i:3d}. {path.name}")

    while True:
        raw = input(f"Select dump index [1-{len(dump_files)}] or 'q': ").strip().lower()
        if raw == "q":
            raise SystemExit(0)
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(dump_files):
                return dump_files[index - 1]
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
    dump_dir_override: str | None = None
    dump_selector: str | None = None
    preview = 8

    dump_dir = resolve_dump_dir(config_path, dump_dir_override)
    dump_files = list_dump_files(dump_dir)
    dump_file = resolve_dump_file(dump_selector, dump_files)

    print(f"Dump directory: {dump_dir}")
    print(f"Selected dump: {dump_file.name}")

    with h5py.File(dump_file, "r") as handle:
        rows = collect_datasets(handle)
        if not rows:
            print("This file contains no datasets.")
            return
        print_summary(rows)
        interactive_dataset_view(rows, preview=preview)


if __name__ == "__main__":
    main()
