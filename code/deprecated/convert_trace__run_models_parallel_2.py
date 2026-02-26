import argparse
import os
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))         # for sibling module import
sys.path.insert(0, str(Path(__file__).parent.parent))  # for config
from config import BASE_PATH
from beckers.Chemistry.code.deprecated.convert_trace__run_models import (
    select_particle_ids,
    convert_phys,
    write_file_params,
    run_model,
    convert_outmodel_to_evolve_output,
    write_analysis_params,
    load_analysis_params,
    find_grid_point,
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
    parser.add_argument("--chemistry", default="Crich", choices=["Crich", "Orich"])

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

    parser.add_argument(
        "--analysis", action="store_true", default=False,
        help="Enable per-particle analysis using an analysis_params.txt file.",
    )
    parser.add_argument(
        "--analysis-params-file", type=str, default=None,
        help="Path to analysis_params.txt. Defaults to <model-base>/param/analysis_params.txt.",
    )
    parser.add_argument(
        "--write-analysis-template", action="store_true", default=False,
        help="Write an analysis_params.txt template pre-filled with selected particle IDs and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    chemistry_type = args.chemistry

    base_path = BASE_PATH
    model_root = base_path / args.model_dir
    model_base = model_root / chemistry_type
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

    # ------------------------------------------------------------------ analysis
    analysis_params = None
    if args.analysis or args.write_analysis_template:
        analysis_params_path = (
            Path(args.analysis_params_file)
            if args.analysis_params_file
            else model_base / "param" / "analysis_params.txt"
        )
        if args.write_analysis_template:
            if analysis_params_path.exists():
                print(f"Analysis params file already exists: {analysis_params_path}")
            else:
                write_analysis_params(selected_ids, analysis_params_path)
                print(f"Edit {analysis_params_path} to enable per-particle analysis, then re-run.")
            return
        if not analysis_params_path.exists():
            raise FileNotFoundError(
                f"Analysis params file not found: {analysis_params_path}\n"
                "Run with --write-analysis-template to create a template."
            )
        analysis_params = load_analysis_params(analysis_params_path)

    print(f"Running {args.model_dir}/{chemistry_type} for {len(target_ids)} particle(s): {target_ids}")
    for particle_id in tqdm(target_ids, total=len(target_ids)):
        input_file = trace_dir / f"{particle_id}.phys"
        if not input_file.exists():
            raise FileNotFoundError(f"Missing trace file: {input_file}")

        output_file = model_base / "input" / f"{particle_id}.txt"
        convert_phys(input_file, output_file)

        if analysis_params is not None and particle_id in analysis_params:
            ana_flag, r_coord = analysis_params[particle_id]
            iana = find_grid_point(model_base, particle_id, r_coord) if ana_flag else 0
            file_params_path = write_file_params(
                model_root, model_base, particle_id,
                chemistry_type=chemistry_type, ana=ana_flag, iana=iana,
            )
        else:
            file_params_path = write_file_params(
                model_root, model_base, particle_id, chemistry_type=chemistry_type
            )

        run_model(model_root, file_params_path)
        convert_outmodel_to_evolve_output(model_root, model_base, particle_id, chemistry_type=chemistry_type, to_cm=args.to_cm)


if __name__ == "__main__":
    main()
