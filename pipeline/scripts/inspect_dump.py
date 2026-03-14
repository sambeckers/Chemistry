#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from common import PipelineConfigManager


class InteractiveHDF5Inspector:
    """Interactive terminal inspector for HDF5 dataset trees."""

    def __init__(self, preview: int = 8) -> None:
        """Set preview size used when displaying flattened dataset values."""
        self.preview = preview

    @staticmethod
    def collect_datasets(handle: h5py.File) -> list[tuple[str, h5py.Dataset]]:
        """Return all datasets in lexical path order."""
        rows: list[tuple[str, h5py.Dataset]] = []

        def visitor(name: str, obj: h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                rows.append((name, obj))

        handle.visititems(visitor)
        rows.sort(key=lambda item: item[0])
        return rows

    @staticmethod
    def _format_shape(shape: tuple[int, ...]) -> str:
        """Format tuple shape as compact x-separated text."""
        return "x".join(str(dim) for dim in shape) if shape else "scalar"

    def print_summary(self, rows: list[tuple[str, h5py.Dataset]]) -> None:
        """Print compact table of dataset paths, shapes, and dtypes."""
        print("\nDataset Summary")
        print("=" * 80)
        print(f"Total datasets: {len(rows)}")
        print(f"{'#':>4}  {'Path':<55} {'Shape':<12} {'DType':<12}")
        print("-" * 80)

        for idx, (name, dataset) in enumerate(rows, start=1):
            shape = self._format_shape(dataset.shape)
            dtype = str(dataset.dtype)
            short_name = name if len(name) <= 55 else f"...{name[-52:]}"
            print(f"{idx:4d}  {short_name:<55} {shape:<12} {dtype:<12}")

    def _preview_array(self, arr: np.ndarray) -> str:
        """Preview first N flattened values while preserving total size info."""
        if arr.size == 0:
            return "[]"
        flat = np.ravel(arr)
        shown = flat[: self.preview]
        values = np.array2string(shown, precision=6, separator=", ", threshold=self.preview)
        if flat.size > self.preview:
            return f"{values} ... (total {flat.size} values)"
        return values

    @staticmethod
    def _safe_stats(arr: np.ndarray) -> str:
        """Compute robust quick stats for numeric and categorical arrays."""
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

    def interactive_dataset_view(self, rows: list[tuple[str, h5py.Dataset]]) -> None:
        """Run REPL loop to inspect one dataset at a time."""
        print("\nInspect datasets")
        print("Type a dataset index to preview it, 'r' to reprint summary, or 'q' to quit.")

        while True:
            raw = input("dataset> ").strip().lower()
            if raw == "q":
                return
            if raw == "r":
                self.print_summary(rows)
                continue
            if not raw.isdigit():
                print("Enter an integer index, 'r', or 'q'.")
                continue

            idx = int(raw)
            if not (1 <= idx <= len(rows)):
                print(f"Index out of range (1-{len(rows)}).")
                continue

            name, dataset = rows[idx - 1]
            arr = np.asarray(dataset[()])
            print("-" * 80)
            print(f"Path: /{name}")
            print(f"Shape: {dataset.shape}")
            print(f"DType: {dataset.dtype}")
            print(f"Stats: {self._safe_stats(arr)}")
            print(f"Preview: {self._preview_array(arr)}")


class DumpInspectorApp:
    """Resolve dump file choice and launch interactive inspection."""

    def __init__(self, config_path: str, dump_dir_arg: str | None, dump_selector: str | None, preview: int) -> None:
        self.config_path = config_path
        self.dump_dir_arg = dump_dir_arg
        self.dump_selector = dump_selector
        self.inspector = InteractiveHDF5Inspector(preview=preview)

    def resolve_dump_dir(self) -> Path:
        """Resolve dump directory from override or config paths.final_output_dir."""
        if self.dump_dir_arg:
            return Path(self.dump_dir_arg).expanduser().resolve()

        config = PipelineConfigManager.load(self.config_path)
        final_output_dir = config.get("paths", {}).get("final_output_dir")
        if not final_output_dir:
            raise KeyError("paths.final_output_dir is missing from config")
        return Path(final_output_dir).expanduser().resolve()

    @staticmethod
    def list_dump_files(dump_dir: Path) -> list[Path]:
        """List available dump_*.h5 files in sorted order."""
        files = sorted(dump_dir.glob("dump_*.h5"))
        if not files:
            raise FileNotFoundError(f"No dump_*.h5 files found in {dump_dir}")
        return files

    @staticmethod
    def _choose_file_interactively(files: list[Path], label: str) -> Path:
        """Present a compact index menu and return selected file path."""
        print(f"Available {label}s:")
        if len(files) <= 30:
            for idx, path in enumerate(files, start=1):
                print(f"  {idx:3d}. {path.name}")
        else:
            head = files[:10]
            tail = files[-10:]
            for idx, path in enumerate(head, start=1):
                print(f"  {idx:3d}. {path.name}")
            print("  ...")
            offset = len(files) - 10
            for i, path in enumerate(tail, start=1):
                print(f"  {offset + i:3d}. {path.name}")

        while True:
            raw = input(f"Select {label} index [1-{len(files)}] or 'q': ").strip().lower()
            if raw == "q":
                raise SystemExit(0)
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(files):
                    return files[index - 1]
            print("Invalid selection. Try again.")

    def resolve_dump_file(self, dump_files: list[Path]) -> Path:
        """Resolve selector text into one concrete dump file path."""
        if not self.dump_selector:
            return self._choose_file_interactively(dump_files, "dump")

        selector = self.dump_selector.strip()
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
            return self._choose_file_interactively(candidates, "dump")

        raise FileNotFoundError(f"Could not resolve dump selector '{self.dump_selector}'")

    def run(self) -> None:
        """Open selected dump and start interactive dataset explorer."""
        dump_dir = self.resolve_dump_dir()
        dump_files = self.list_dump_files(dump_dir)
        dump_file = self.resolve_dump_file(dump_files)

        print(f"Dump directory: {dump_dir}")
        print(f"Selected dump: {dump_file.name}")

        with h5py.File(dump_file, "r") as handle:
            rows = self.inspector.collect_datasets(handle)
            if not rows:
                print("This file contains no datasets.")
                return
            self.inspector.print_summary(rows)
            self.inspector.interactive_dataset_view(rows)


def main() -> None:
    """Editable entrypoint; update variables below before running."""
    # Edit these variables directly before running the script.
    config_path = str(Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml")
    dump_dir_override: str | None = None
    dump_selector: str | None = None
    preview = 8

    app = DumpInspectorApp(
        config_path=config_path,
        dump_dir_arg=dump_dir_override,
        dump_selector=dump_selector,
        preview=preview,
    )
    app.run()


if __name__ == "__main__":
    main()
