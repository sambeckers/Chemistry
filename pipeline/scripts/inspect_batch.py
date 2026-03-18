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

        if arr.dtype.kind in {"S", "U"}:
            uniques = np.unique(np.ravel(arr))
            return f"unique={len(uniques)}"

        if arr.dtype.kind == "O":
            flat = np.ravel(arr)
            try:
                uniques = np.unique(flat)
                return f"unique={len(uniques)}"
            except Exception:
                # Object arrays can contain nested arrays that are not safely orderable.
                try:
                    normalized = [repr(value) for value in flat]
                    return f"unique={len(set(normalized))}"
                except Exception:
                    return "stats unavailable"

        return "stats unavailable"

    @staticmethod
    def _find_particle_id_dataset(rows: list[tuple[str, h5py.Dataset]]) -> tuple[str, h5py.Dataset] | None:
        """Find particle-ID dataset, preferring canonical paths."""
        preferred_paths = ["particles/id", "trace/particles/id", "id", "114"]
        for preferred in preferred_paths:
            matches = [(name, dataset) for name, dataset in rows if name == preferred]
            if len(matches) == 1:
                return matches[0]

        by_leaf = [(name, dataset) for name, dataset in rows if name.split("/")[-1] in {"id", "114"}]
        if len(by_leaf) == 1:
            return by_leaf[0]
        return None

    @staticmethod
    def _match_particle_ids(particle_ids: np.ndarray, raw_value: str) -> np.ndarray:
        """Return matching index positions for a user-entered particle ID."""
        ids = np.ravel(particle_ids)
        if ids.size == 0:
            return np.array([], dtype=np.int64)

        if np.issubdtype(ids.dtype, np.integer):
            try:
                target = int(raw_value)
            except ValueError:
                return np.array([], dtype=np.int64)
            return np.where(ids == target)[0]

        if np.issubdtype(ids.dtype, np.floating):
            try:
                target = float(raw_value)
            except ValueError:
                return np.array([], dtype=np.int64)
            return np.where(np.isclose(ids, target, equal_nan=True))[0]

        normalized = np.array([value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value) for value in ids])
        return np.where(normalized == raw_value)[0]

    def _interactive_particle_view(
        self,
        dataset_name: str,
        dataset_arr: np.ndarray,
        particle_ids: np.ndarray | None,
    ) -> str:
        """Inspect one dataset entry at a time by index, and by particle ID when available."""
        ids = np.ravel(particle_ids) if particle_ids is not None else None
        supports_particle_id = ids is not None and dataset_arr.ndim > 0 and dataset_arr.shape[0] == ids.size
        item_count = int(dataset_arr.shape[0]) if dataset_arr.ndim > 0 else 1

        print("-" * 80)
        print(f"Path: /{dataset_name}")
        print("Select one entry to inspect.")
        if supports_particle_id:
            print("Commands: particle ID value, 'i <index>', 'l' (IDs sample), 'li' (index range), 'b', 'q'.")
        else:
            print("Commands: 'i <index>' (or index), 'li' (index range), 'b', 'q'.")

        while True:
            raw = input("particle-id> ").strip()
            lowered = raw.lower()
            if lowered == "q":
                return "quit"
            if lowered == "b":
                return "back"
            if lowered == "li":
                if item_count == 0:
                    print("No entries in this dataset.")
                else:
                    max_idx = item_count - 1
                    print(f"Valid index range: 0..{max_idx}")
                continue
            if lowered == "l":
                if supports_particle_id and ids is not None:
                    shown = ids[: self.preview]
                    print(f"Particle ID sample: {np.array2string(shown, separator=', ')}")
                    if ids.size > self.preview:
                        print(f"(showing {self.preview} of {ids.size})")
                else:
                    print("Particle-ID lookup is unavailable for this dataset; use index selection.")
                continue

            selected_index: int | None = None

            if lowered.startswith("i "):
                token = raw[2:].strip()
                if not token:
                    print("Provide an index after 'i', for example: i 3")
                    continue
                try:
                    selected_index = int(token)
                except ValueError:
                    print("Index must be an integer.")
                    continue
            elif raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
                if supports_particle_id:
                    matches = self._match_particle_ids(ids, raw) if ids is not None else np.array([], dtype=np.int64)
                    if matches.size == 0:
                        selected_index = int(raw)
                    else:
                        selected_index = int(matches[0])
                        if matches.size > 1:
                            print(
                                f"Found {matches.size} matches for particle ID '{raw}'. "
                                f"Using first index {selected_index}."
                            )
                else:
                    selected_index = int(raw)
            else:
                if supports_particle_id and ids is not None:
                    matches = self._match_particle_ids(ids, raw)
                    if matches.size == 0:
                        print(f"Particle ID '{raw}' not found. Use 'i <index>' for direct index selection.")
                        continue
                    selected_index = int(matches[0])
                    if matches.size > 1:
                        print(
                            f"Found {matches.size} matches for particle ID '{raw}'. "
                            f"Using first index {selected_index}."
                        )
                else:
                    print("Enter 'i <index>' (or an integer index), 'li', 'b', or 'q'.")
                    continue

            if dataset_arr.ndim == 0:
                selected_index = 0

            if selected_index is None or selected_index < 0 or selected_index >= item_count:
                max_idx = item_count - 1
                print(f"Index out of range (0-{max_idx}).")
                continue

            particle_data = np.asarray(dataset_arr[selected_index]) if dataset_arr.ndim > 0 else np.asarray(dataset_arr)
            print("-" * 80)
            print(f"Path: /{dataset_name}")
            if supports_particle_id and ids is not None:
                particle_id_value = ids[selected_index]
                print(f"Particle ID: {particle_id_value} (index {selected_index})")
            else:
                print(f"Index: {selected_index}")
            print(f"Particle data shape: {particle_data.shape}")
            print(f"Particle data dtype: {particle_data.dtype}")
            print(f"Stats: {self._safe_stats(particle_data)}")
            print(f"Preview: {self._preview_array(particle_data)}")
            print("Type 'b' to go back to selection, or 'q' to quit.")

            while True:
                view_cmd = input("particle-view> ").strip().lower()
                if view_cmd == "q":
                    return "quit"
                if view_cmd in {"", "b"}:
                    break
                print("Enter 'b' (or Enter) to pick another entry, or 'q' to quit.")

    def interactive_dataset_view(self, rows: list[tuple[str, h5py.Dataset]]) -> str:
        """Run REPL loop to inspect one dataset at a time.

        Returns:
            "back" to go back to batch selection.
            "quit" to exit the inspector.
        """
        particle_id_entry = self._find_particle_id_dataset(rows)
        particle_ids: np.ndarray | None = None
        particle_id_name: str | None = None
        if particle_id_entry is not None:
            particle_id_name, particle_id_dataset = particle_id_entry
            particle_ids = np.ravel(np.asarray(particle_id_dataset[()]))

        print("\nInspect datasets")
        print("Type a dataset index to inspect it, 'r' to reprint summary, 'b' to go back, or 'q' to quit.")
        if particle_ids is not None:
            print("For non-ID datasets, you can inspect by particle ID or index.")

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

            if particle_ids is not None and particle_id_name is not None and name != particle_id_name:
                print("-" * 80)
                print(f"Path: /{name}")
                print(f"Shape: {dataset.shape}")
                print(f"DType: {dataset.dtype}")
                print(f"Stats: {self._safe_stats(arr)}")
                print(f"Preview: {self._preview_array(arr)}")

                if arr.ndim == 0:
                    print("This dataset is scalar and cannot be opened in particle/item selection mode.")
                    continue

                supports_particle_id = arr.shape[0] == particle_ids.size
                if supports_particle_id:
                    print("Type 'p' to enter particle selection (ID/index), or Enter to return to dataset list.")
                else:
                    print("Type 'p' to enter item selection by index, or Enter to return to dataset list.")

                while True:
                    view_cmd = input("view> ").strip().lower()
                    if view_cmd == "q":
                        return "quit"
                    if view_cmd in {"", "b"}:
                        break
                    if view_cmd == "p":
                        action = self._interactive_particle_view(
                            name,
                            arr,
                            particle_ids if supports_particle_id else None,
                        )
                        if action == "quit":
                            return "quit"
                        break
                    print("Enter 'p' to open selection mode, Enter to continue, or 'q' to quit.")
                continue

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
