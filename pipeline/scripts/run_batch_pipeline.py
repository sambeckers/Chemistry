from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from append_ev_to_hdf5 import append_particle_to_batch, initialise_batch_file
from common import (
    PIPELINE_ROOT,
    ensure_clean_directory,
    get_batch_particle_ids,
    get_selected_dump_numbers,
    get_species_filename,
    get_species_for_chemistry,
    get_target_dumps,
    load_pipeline_config,
)
from convert_phys_to_txt import convert_phys_file


FORTRAN_PATH_LIMIT = 70


def write_trace_config(config_path: Path, particle_ids) -> None:
    config_path.write_text(
        "&trace_config\n"
        f"  id_start = {int(particle_ids[0])}\n"
        f"  id_end = {int(particle_ids[-1])}\n"
        "  n_boundary = 0\n"
        "/\n",
        encoding="ascii",
    )


def ensure_link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    os.symlink(source, destination)


def prepare_trace_workspace(config: dict, batch_work_dir: Path, particle_ids) -> Path:
    phantomanalysis_binary = Path(config["paths"]["phantomanalysis_binary"]).resolve()
    av_dir = Path(config["paths"]["av_dir"]).resolve()

    ensure_clean_directory(batch_work_dir)
    trace_output_dir = ensure_clean_directory(batch_work_dir / "trace_output")
    write_trace_config(batch_work_dir / "trace.cfg", particle_ids)
    (batch_work_dir / "batch_particle_ids.json").write_text(
        json.dumps([int(pid) for pid in particle_ids.tolist()]), encoding="ascii"
    )

    copied_binary = batch_work_dir / "phantomanalysis"
    shutil.copy2(phantomanalysis_binary, copied_binary)
    copied_binary.chmod(0o755)
    ensure_link(av_dir, batch_work_dir / "AV")
    return trace_output_dir


def run_trace_stage(config: dict, batch_work_dir: Path) -> None:
    selected_dumps = get_selected_dump_numbers(config)
    if len(selected_dumps) < 2:
        raise RuntimeError("Trace stage requires at least two dumps.")

    binary = batch_work_dir / "phantomanalysis"
    data_dir = Path(config["paths"]["phantom_dump_dir"])
    prefix = config["simulation"]["prefix"]
    dump_paths = [str(data_dir / f"{prefix}_{dump_number:05d}") for dump_number in selected_dumps]
    subprocess.run([str(binary), *dump_paths], cwd=batch_work_dir, check=True)


def all_trace_files_present(trace_output_dir: Path, particle_ids) -> bool:
    for particle_id in particle_ids.tolist():
        if not (trace_output_dir / f"{particle_id}.phys").exists():
            return False
    return True


def chemistry_runtime_dirs(batch_work_dir: Path, chemistry_type: str) -> dict[str, Path]:
    base = ensure_clean_directory(batch_work_dir / "chemistry_runtime" / chemistry_type)
    dirs = {
        "base": base,
        "param": ensure_clean_directory(base / "param" / "trace_file_param"),
        "input": ensure_clean_directory(base / "input"),
        "output": ensure_clean_directory(base / "output"),
        "ev_output": ensure_clean_directory(base / "ev_output"),
        "analyse_output": ensure_clean_directory(base / "analyse_output"),
    }
    return dirs


def _assert_fortran_path_length(value: str, label: str) -> None:
    if len(value) > FORTRAN_PATH_LIMIT:
        raise ValueError(
            f"{label} path is too long for Fortran A70 reader ({len(value)} > {FORTRAN_PATH_LIMIT}): {value}"
        )


def prepare_model_runtime_links(
    model_root: Path,
    runtime_dirs: dict[str, Path],
    batch_index: int,
    chemistry_type: str,
) -> Path:
    chem_tag = chemistry_type[:1].lower() if chemistry_type else "x"
    runtime_root = model_root / "_rt" / f"b{batch_index:05d}_{chem_tag}"
    ensure_clean_directory(runtime_root)

    ensure_link(runtime_dirs["input"], runtime_root / "in")
    ensure_link(runtime_dirs["output"], runtime_root / "out")
    ensure_link(runtime_dirs["ev_output"], runtime_root / "ev")
    ensure_link(runtime_dirs["analyse_output"], runtime_root / "ana")
    ensure_link(runtime_dirs["param"], runtime_root / "par")
    return runtime_root.relative_to(model_root)


def write_file_params(
    model_root: Path,
    runtime_dirs: dict[str, Path],
    model_runtime_rel: Path,
    particle_id: int,
    chemistry_type: str,
) -> Path:
    file_params_path = runtime_dirs["param"] / f"file_parameters_{particle_id}.txt"

    phys_file = (model_runtime_rel / "in" / f"{particle_id}.txt").as_posix()
    reac_file = "rate12_complex.rates"
    spec_file = get_species_filename(chemistry_type)
    bind_file = "rate12_binding.dat"
    switch_file = "param/reaction_parameters.txt"
    grain_file = "param/grain_parameters.txt"
    rad_file = "param/radiation_parameters.txt"
    out_file = (model_runtime_rel / "out" / f"model_output_{particle_id}.dat").as_posix()
    rates_file = (model_runtime_rel / "out" / f"model_rates_{particle_id}.dat").as_posix()
    ana_file = (model_runtime_rel / "ana" / f"model_analyse_{particle_id}.dat").as_posix()

    for label, value in (
        ("physical", phys_file),
        ("reaction", reac_file),
        ("species", spec_file),
        ("binding", bind_file),
        ("switch", switch_file),
        ("grain", grain_file),
        ("radiation", rad_file),
        ("out", out_file),
        ("rates", rates_file),
        ("analyse", ana_file),
    ):
        _assert_fortran_path_length(value, label)

    with open(file_params_path, "w", encoding="ascii") as handle:
        handle.write("! FILE PARAMETERS FOR ENVELOPE MODEL\n")
        handle.write("! Physical conditions input file\n")
        handle.write(f"{phys_file}\n")
        handle.write("! Reaction file\n")
        handle.write(f"{reac_file}\n")
        handle.write("! Species file\n")
        handle.write(f"{spec_file}\n")
        handle.write("! Binding energies file\n")
        handle.write(f"{bind_file}\n")
        handle.write("! Reaction parameters file\n")
        handle.write(f"{switch_file}\n")
        handle.write("! Grain parameters file\n")
        handle.write(f"{grain_file}\n")
        handle.write("! Radiation field parameters file\n")
        handle.write(f"{rad_file}\n")
        handle.write("! Output abundances file\n")
        handle.write(f"{out_file}\n")
        handle.write("! Output rates file\n")
        handle.write(f"{rates_file}\n")
        handle.write("! Call analyse subroutine (T(RUE) or F(ALSE))?\n")
        handle.write("F\n")
        handle.write("! Grid point to run analyse\n")
        handle.write("0\n")
        handle.write("! Analyse file\n")
        handle.write(f"{ana_file}\n")
    return file_params_path


def run_chemistry_model(model_root: Path, file_params_arg: str) -> None:
    _assert_fortran_path_length(file_params_arg, "file_parameters")
    subprocess.run([str((model_root / "model").resolve()), file_params_arg], cwd=model_root, check=True)


def convert_model_output(model_root: Path, runtime_dirs: dict[str, Path], particle_id: int, chemistry_type: str) -> Path:
    output_prefix = runtime_dirs["ev_output"] / f"ev_{particle_id}"
    command = [
        "perl",
        str((model_root / "evolve_output.pl").resolve()),
        str((runtime_dirs["output"] / f"model_output_{particle_id}.dat").resolve()),
        str(output_prefix.resolve()),
    ]
    command.extend(get_species_for_chemistry(chemistry_type))
    subprocess.run(command, cwd=model_root, check=True)
    ev_output_path = output_prefix.with_suffix(".dat")
    postprocess_ev_output(ev_output_path, to_cm=True)
    return ev_output_path


def postprocess_ev_output(filepath: Path, to_cm: bool = True) -> None:
    if not filepath.exists():
        return

    raw_lines = filepath.read_text(encoding="ascii").splitlines(keepends=True)
    if len(raw_lines) < 6:
        return

    names_idx = 4
    units_idx = 5

    col_names = raw_lines[names_idx].split()
    if "X" not in col_names or "Y" not in col_names or "Z" not in col_names:
        return

    xi = col_names.index("X")
    yi = col_names.index("Y")
    zi = col_names.index("Z")

    if "R" in col_names:
        ri_existing = col_names.index("R")
    else:
        ri_existing = -1

    pc_to_cm = 3.085677581491367e18
    unit_str = "(CM)" if to_cm else "(PC)"

    if ri_existing < 0:
        new_col_names = col_names[: zi + 1] + ["R"] + col_names[zi + 1 :]
    else:
        new_col_names = col_names[:]
    names_line = "  ".join(f"{name:<16}" for name in new_col_names) + "\n"

    data_start = units_idx
    if units_idx < len(raw_lines) and raw_lines[units_idx].lstrip().startswith("#"):
        old_units = raw_lines[units_idx].lstrip("#").split()
        if ri_existing < 0:
            new_units = old_units[: zi + 1] + [unit_str] + old_units[zi + 1 :]
        else:
            new_units = old_units[:]
        if to_cm:
            new_units[xi] = unit_str
            new_units[yi] = unit_str
            new_units[zi] = unit_str
            if ri_existing >= 0:
                new_units[ri_existing] = unit_str
            else:
                new_units[zi + 1] = unit_str
        units_line = "#  " + "  ".join(f"{unit:<16}" for unit in new_units) + "\n"
        data_start = units_idx + 1
    else:
        units_line = raw_lines[units_idx]

    new_lines = raw_lines[:names_idx] + [names_line, units_line]

    for line in raw_lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        parts = stripped.split()
        x = float(parts[xi])
        y = float(parts[yi])
        z = float(parts[zi])
        if to_cm:
            x *= pc_to_cm
            y *= pc_to_cm
            z *= pc_to_cm
            parts[xi] = f"{x:.5e}"
            parts[yi] = f"{y:.5e}"
            parts[zi] = f"{z:.5e}"

        r = np.sqrt(x * x + y * y + z * z)
        if ri_existing >= 0:
            parts[ri_existing] = f"{r:.5e}"
        else:
            parts = parts[: zi + 1] + [f"{r:.5e}"] + parts[zi + 1 :]

        new_lines.append("  ".join(parts) + "\n")

    filepath.write_text("".join(new_lines), encoding="ascii")


def cleanup_particle_products(
    trace_file: Path | None,
    runtime_dirs: dict[str, Path],
    particle_id: int,
    cleanup_trace_file: bool,
    cleanup_ev_output: bool,
) -> None:
    candidates = [
        runtime_dirs["input"] / f"{particle_id}.txt",
        runtime_dirs["output"] / f"model_output_{particle_id}.dat",
        runtime_dirs["output"] / f"model_rates_{particle_id}.dat",
        runtime_dirs["analyse_output"] / f"model_analyse_{particle_id}.dat",
        runtime_dirs["param"] / f"file_parameters_{particle_id}.txt",
    ]
    if cleanup_ev_output:
        candidates.append(runtime_dirs["ev_output"] / f"ev_{particle_id}.dat")
    if cleanup_trace_file and trace_file is not None:
        candidates.append(trace_file)

    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()


def process_particle_batch(config: dict, batch_index: int) -> Path:
    particle_ids, metadata = get_batch_particle_ids(config, batch_index)
    if len(particle_ids) == 0:
        raise RuntimeError(f"Batch {batch_index} contains no particles")

    compression = config["processing"].get("compression", "gzip")
    compression_level = config["processing"].get("compression_level", 4)
    chemistry_types = list(config["processing"].get("chemistry_types", ["Crich"]))
    dump_numbers = get_target_dumps(config)

    batch_output_dir = ensure_clean_directory(config["paths"]["batch_output_dir"])
    scratch_root = ensure_clean_directory(config["paths"]["scratch_root"])
    batch_file = batch_output_dir / f"batch_{batch_index:05d}.h5"
    batch_work_dir = scratch_root / f"batch_{batch_index:05d}"

    overwrite_batch = bool(config["processing"].get("overwrite_batch", False))
    reuse_trace_output = bool(config["processing"].get("reuse_existing_trace_output", False))
    if batch_file.exists():
        if not overwrite_batch:
            raise FileExistsError(f"Batch output already exists: {batch_file}")
        batch_file.unlink()

    if batch_work_dir.exists():
        if not reuse_trace_output:
            shutil.rmtree(batch_work_dir)

    print(
        f"Batch {batch_index}: particles {metadata['selection_start']}:{metadata['selection_end']} "
        f"({len(particle_ids)} particle(s) out of {metadata['total_particles']})"
    )

    trace_output_dir = prepare_trace_workspace(config, batch_work_dir, particle_ids)
    if reuse_trace_output and all_trace_files_present(trace_output_dir, particle_ids):
        print(f"Batch {batch_index}: reusing existing trace_output (skip phantomanalysis)")
    else:
        if reuse_trace_output:
            raise RuntimeError(
                "reuse_existing_trace_output is true, but trace_output is incomplete for this batch. "
                "Disable reuse_existing_trace_output or rebuild trace_output for this batch."
            )
        run_trace_stage(config, batch_work_dir)

    initialise_batch_file(batch_file, batch_index, particle_ids, dump_numbers, chemistry_types)

    model_root = Path(config["paths"]["chemistry_model_root"]).resolve()
    cleanup_temporary = bool(config["processing"].get("cleanup_temporary", True))
    reuse_existing_ev_output = bool(config["processing"].get("reuse_existing_ev_output", True))

    runtime_dirs_by_type: dict[str, dict[str, Path]] = {}
    runtime_rel_by_type: dict[str, Path] = {}
    for chemistry_type in chemistry_types:
        runtime_dirs = chemistry_runtime_dirs(batch_work_dir, chemistry_type)
        runtime_dirs_by_type[chemistry_type] = runtime_dirs
        runtime_rel_by_type[chemistry_type] = prepare_model_runtime_links(
            model_root=model_root,
            runtime_dirs=runtime_dirs,
            batch_index=batch_index,
            chemistry_type=chemistry_type,
        )

    for slot, particle_id in enumerate(particle_ids.tolist()):
        trace_file = trace_output_dir / f"{particle_id}.phys"
        if not trace_file.exists():
            raise FileNotFoundError(f"Missing trace file for particle {particle_id}: {trace_file}")

        for chemistry_type in chemistry_types:
            runtime_dirs = runtime_dirs_by_type[chemistry_type]
            model_runtime_rel = runtime_rel_by_type[chemistry_type]
            input_file = runtime_dirs["input"] / f"{particle_id}.txt"
            ev_output_path = runtime_dirs["ev_output"] / f"ev_{particle_id}.dat"

            if reuse_existing_ev_output and ev_output_path.exists() and ev_output_path.stat().st_size > 0:
                print(
                    f"Batch {batch_index}: reusing existing ev_output for particle {particle_id} ({chemistry_type})"
                )
            else:
                convert_phys_file(trace_file, input_file)
                file_params_path = write_file_params(model_root, runtime_dirs, model_runtime_rel, particle_id, chemistry_type)
                file_params_arg = (model_runtime_rel / "par" / file_params_path.name).as_posix()
                run_chemistry_model(model_root, file_params_arg)
                ev_output_path = convert_model_output(model_root, runtime_dirs, particle_id, chemistry_type)

            append_particle_to_batch(
                batch_file=batch_file,
                slot=slot,
                particle_id=particle_id,
                chemistry_type=chemistry_type,
                ev_output_path=ev_output_path,
                dump_numbers=dump_numbers,
                compression=compression,
                compression_level=compression_level,
            )
            cleanup_particle_products(
                trace_file,
                runtime_dirs,
                particle_id,
                cleanup_trace_file=False,
                cleanup_ev_output=not reuse_existing_ev_output,
            )

        if cleanup_temporary and trace_file.exists():
            trace_file.unlink()

    if cleanup_temporary:
        for model_runtime_rel in runtime_rel_by_type.values():
            link_root = model_root / model_runtime_rel
            if link_root.exists():
                shutil.rmtree(link_root)
        shutil.rmtree(batch_work_dir)

    return batch_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one batch of the standalone HDF5 chemistry pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    batch_file = process_particle_batch(config, args.batch_index)
    print(f"Completed batch {args.batch_index}: {batch_file}")


if __name__ == "__main__":
    main()
