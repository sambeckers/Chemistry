import argparse
import os
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))         # for sibling module import
sys.path.insert(0, str(Path(__file__).parent.parent))  # for config
from config import BASE_PATH
from convert_trace__run_models import (
    select_particle_ids,
    convert_phys,
    write_file_params,
    postprocess_ev_output,
    run_model,
    convert_outmodel_to_evolve_output,
)


def apply_custom_particle_overrides(particle_ids):
    """Mirror custom particle substitutions from the original script."""
    if len(particle_ids) > 1:
        particle_ids[1] = 29823
    if len(particle_ids) > 2:
        particle_ids[2] = 46371
    return particle_ids


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
    parser.add_argument(
        "--no-cm", dest="to_cm", action="store_false",
        help="Keep X/Y/Z/R in pc instead of converting to cm (default: convert to cm)",
    )
    parser.set_defaults(to_cm=True)
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
        convert_outmodel_to_evolve_output(model_base, particle_id, to_cm=args.to_cm)


if __name__ == "__main__":
    main()
