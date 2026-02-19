import argparse
import os
from pathlib import Path
import sys

import numpy as np
from astropy import units as u
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_PATH


def select_particle_ids(
    particle_ids_file,
    trace_dir,
    max_particle_id=300000,
    start_index=2,
    n_select=15,
):
    with open(particle_ids_file, "r") as f:
        particle_ids = [int(line.strip()) for line in f if line.strip()]

    particle_ids = [pid for pid in particle_ids if pid <= max_particle_id]
    particle_ids = [pid for pid in particle_ids if (trace_dir / f"{pid}.phys").exists()]

    if not particle_ids:
        return []

    n_pick = min(n_select, len(particle_ids))
    indices = np.linspace(start_index, len(particle_ids) - 1, num=n_pick, dtype=int)
    return [particle_ids[i] for i in indices]


def apply_custom_particle_overrides(particle_ids):
    """Mirror custom particle substitutions from the original script."""
    if len(particle_ids) > 1:
        particle_ids[1] = 29823
    if len(particle_ids) > 2:
        particle_ids[2] = 46371
    return particle_ids


def convert_phys(in_file, out_file):
    rows = []
    with open(in_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            values = line.split()
            rows.append(
                {
                    "time_str": values[0],
                    "x": float(values[1]),
                    "y": float(values[2]),
                    "z": float(values[3]),
                    "dens_str": values[4],
                    "temp_str": values[5],
                    "av_str": values[7],
                }
            )

    with open(out_file, "w") as f:
        for row in rows:
            x_pc = (row["x"] * u.AU).to(u.pc).value
            y_pc = (row["y"] * u.AU).to(u.pc).value
            z_pc = (row["z"] * u.AU).to(u.pc).value

            line = (
                f"    {x_pc:.14e}"
                f"    {y_pc:.14e}"
                f"    {z_pc:.14e}"
                f"    {row['dens_str']}"
                f"    {row['temp_str']}"
                f"    {row['temp_str']}"
                f"    {row['av_str']}"
                f"    0.00000e+00"
                f"    0.00000e+00"
                f"    {row['time_str']}\n"
            )
            f.write(line)


def write_file_params(model_base, particle_id):
    param_file = model_base / "param" / "trace_file_param" / f"file_parameters_{particle_id}.txt"
    with open(param_file, "w") as fpfile:
        fpfile.write("! FILE PARAMETERS FOR ENVELOPE MODEL\n")
        fpfile.write("! Physical conditions input file\n")
        fpfile.write(f"input/{particle_id}.txt\n")
        fpfile.write("! Reaction file\n")
        fpfile.write("./rate12_complex.rates\n")
        fpfile.write("! Species file\n")
        fpfile.write("./rate12_complex_atomic_Crich.specs\n")
        fpfile.write("! Binding energies file\n")
        fpfile.write("./rate12_binding.dat\n")
        fpfile.write("! Reaction parameters file\n")
        fpfile.write("param/reaction_parameters.txt\n")
        fpfile.write("! Grain parameters file\n")
        fpfile.write("param/grain_parameters.txt\n")
        fpfile.write("! Radiation field parameters file\n")
        fpfile.write("param/radiation_parameters.txt\n")
        fpfile.write("! Output abundances file\n")
        fpfile.write(f"output/model_output_{particle_id}.dat\n")
        fpfile.write("! Output rates file\n")
        fpfile.write(f"output/model_rates_{particle_id}.dat\n")
        fpfile.write("! Call analyse subroutine (T(RUE) or F(ALSE))?\n")
        fpfile.write("F\n")
        fpfile.write("! Grid point to run analyse\n")
        fpfile.write("0\n")
        fpfile.write("! Analyse file\n")
        fpfile.write(f"./model_analyse_{particle_id}.dat\n")


def run_model(model_base, particle_id):
    os.system(
        f"cd {model_base} && "
        f"time ./model param/trace_file_param/file_parameters_{particle_id}.txt"
    )


def convert_outmodel_to_evolve_output(model_base, particle_id):
    parents = ["CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN", "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS"]
    daughters = ["CN", "C2H", "C4H", "C6H", "HC3N", "HC5N", "HC7N"]
    daughters_2 = ["HCO+", "CH", "CH2", "CH3", "NH", "NH2", "NH3", "SO", "SO2", "HS", "HCNH+", "NH4+"]
    daughters_3 = ["H2CO", "H2CS", "CH3CN", "SiC", "SiN"]
    grains = ["GSiO", "GH2O", "GC2H2", "GHCN", "GH2S", "GCH4", "GC2H4"]
    atoms = ["C", "N", "H", "O", "S", "Si", "Cl", "F", "P"]
    atoms_plus = ["C+", "N+", "H+", "O+", "S+", "Si+", "Cl+", "F+", "P+"]

    molecules = parents + daughters + daughters_2 + daughters_3 + grains + atoms + atoms_plus
    molecule_str = " ".join(molecules)

    os.system(
        f"cd {model_base} && "
        f"perl evolve_output.pl output/model_output_{particle_id}.dat ev_output/ev_{particle_id} {molecule_str}"
    )


def ensure_directories(model_base):
    for dir_name in ["param/trace_file_param", "input", "output", "ev_output"]:
        (model_base / dir_name).mkdir(parents=True, exist_ok=True)


def resolve_target_particle(args, selected_ids):
    if args.particle_id is not None:
        return [args.particle_id]

    if args.use_slurm_array:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_id is not None:
            idx = int(array_id) - args.slurm_index_base
            if idx < 0 or idx >= len(selected_ids):
                raise IndexError(
                    f"SLURM_ARRAY_TASK_ID={array_id} maps to index {idx}, "
                    f"but valid index range is 0..{len(selected_ids)-1}"
                )
            return [selected_ids[idx]]

    return selected_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert trace files and run chemistry model, optionally one particle per task."
    )
    parser.add_argument("--pmf", default="wind_v10")
    parser.add_argument("--traces-subdir", default="trace_output_with_av")
    parser.add_argument("--model-dir", default="evolving_model")

    parser.add_argument("--max-particle-id", type=int, default=300000)
    parser.add_argument("--start-index", type=int, default=2)
    parser.add_argument("--n-select", type=int, default=15)

    parser.add_argument("--particle-id", type=int, default=None)
    parser.add_argument("--use-slurm-array", action="store_true")
    parser.add_argument("--slurm-index-base", type=int, default=0)
    parser.add_argument("--selected-ids-out", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    base_path = BASE_PATH
    model_base = base_path / args.model_dir
    trace_dir = base_path / "traces" / args.pmf / args.traces_subdir
    particle_ids_file = base_path / "traces" / args.pmf / "particle_IDs.txt"

    selected_ids = select_particle_ids(
        particle_ids_file,
        trace_dir,
        max_particle_id=args.max_particle_id,
        start_index=args.start_index,
        n_select=args.n_select,
    )
    selected_ids = apply_custom_particle_overrides(selected_ids)

    if not selected_ids and args.particle_id is None:
        raise RuntimeError("No valid particle IDs found to process.")

    if args.selected_ids_out:
        out_path = Path(args.selected_ids_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for pid in selected_ids:
                f.write(f"{pid}\n")

    target_ids = resolve_target_particle(args, selected_ids)

    ensure_directories(model_base)

    print(f"Running {args.model_dir} for {len(target_ids)} particle(s): {target_ids}")
    for particle_id in tqdm(target_ids, total=len(target_ids)):
        input_file = trace_dir / f"{particle_id}.phys"
        if not input_file.exists():
            raise FileNotFoundError(f"Missing trace file: {input_file}")

        output_file = model_base / "input" / f"{particle_id}.txt"
        convert_phys(input_file, output_file)
        write_file_params(model_base, particle_id)
        run_model(model_base, particle_id)
        convert_outmodel_to_evolve_output(model_base, particle_id)


if __name__ == "__main__":
    main()
