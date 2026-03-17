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

    def interactive_dataset_view(self, rows: list[tuple[str, h5py.Dataset]]) -> str:
        """Run REPL loop to inspect one dataset at a time.

        Returns:
            "back" to go back to batch selection.
            "quit" to exit the inspector.
        """
        print("\nInspect datasets")
        print("Type a dataset index to preview it, 'r' to reprint summary, 'b' to go back, or 'q' to quit.")

        while True:
            raw = input("dataset> ").strip().lower()
            if raw == "q":
                return "quit"
            if raw == "b":
                return "back"
            if raw == "r":
                self.print_summary(rows)
                continue
            if not raw.isdigit():
                print("Enter an integer index, 'r', 'b', or 'q'.")
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


class BatchInspectorApp:
    """Resolve batch file choice and launch interactive inspection."""

    def __init__(self, config_path: str, batch_dir_arg: str | None, batch_selector: str | None, preview: int) -> None:
        self.config_path = config_path
        self.batch_dir_arg = batch_dir_arg
        self.batch_selector = batch_selector
        self.inspector = InteractiveHDF5Inspector(preview=preview)

    def resolve_batch_dir(self) -> Path:
        """Resolve batch directory from override or config paths.batch_output_dir."""
        if self.batch_dir_arg:
            return Path(self.batch_dir_arg).expanduser().resolve()

        config = PipelineConfigManager.load(self.config_path)
        batch_output_dir = config.get("paths", {}).get("batch_output_dir")
        if not batch_output_dir:
            raise KeyError("paths.batch_output_dir is missing from config")
        return Path(batch_output_dir).expanduser().resolve()

    @staticmethod
    def list_batch_files(batch_dir: Path) -> list[Path]:
        """List available batch_*.h5 files in sorted order."""
        files = sorted(batch_dir.glob("batch_*.h5"))
        if not files:
            raise FileNotFoundError(f"No batch_*.h5 files found in {batch_dir}")
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

    def resolve_batch_file(self, batch_files: list[Path]) -> Path:
        """Resolve selector text into one concrete batch file path."""
        if not self.batch_selector:
            return self._choose_file_interactively(batch_files, "batch")

        selector = self.batch_selector.strip()
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
            return self._choose_file_interactively(candidates, "batch")

        raise FileNotFoundError(f"Could not resolve batch selector '{self.batch_selector}'")

    def run(self) -> None:
        """Open selected batch and start interactive dataset explorer."""
        batch_dir = self.resolve_batch_dir()
        batch_files = self.list_batch_files(batch_dir)
        force_interactive_selection = False

        while True:
            if force_interactive_selection:
                batch_file = self._choose_file_interactively(batch_files, "batch")
            else:
                batch_file = self.resolve_batch_file(batch_files)

            print(f"Batch directory: {batch_dir}")
            print(f"Selected batch: {batch_file.name}")

            with h5py.File(batch_file, "r") as handle:
                rows = self.inspector.collect_datasets(handle)
                if not rows:
                    print("This file contains no datasets.")
                    force_interactive_selection = True
                    continue
                self.inspector.print_summary(rows)
                action = self.inspector.interactive_dataset_view(rows)

            if action == "quit":
                return
            force_interactive_selection = True


def main() -> None:
    """Editable entrypoint; update variables below before running."""
    # Edit these variables directly before running the script.
    config_path = str(Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml")
    batch_dir_override: str | None = None
    batch_selector: str | None = None
    preview = 8

    app = BatchInspectorApp(
        config_path=config_path,
        batch_dir_arg=batch_dir_override,
        batch_selector=batch_selector,
        preview=preview,
    )
    app.run()


if __name__ == "__main__":
    main()
