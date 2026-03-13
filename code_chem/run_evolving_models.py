"""Run chemistry models on phantom particle traces.

Usage
-----
Interactive / local (default):
    python run_chemistry_models.py

SLURM array (one particle per task, set by the slurm script automatically):
    python run_chemistry_models.py --parallel

The --parallel flag makes the script non-interactive: it skips directory-setup
prompts and uses SLURM_ARRAY_TASK_ID to pick the particle for this task.

All science settings (chemistry type, paths, particle selection, analysis
flags, …) live in the CONFIG section at the top of main() below — edit those
instead of the command-line interface.
"""

import argparse
import numpy as np
import os
import shutil
from pathlib import Path
from astropy import units as u
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BASE_PATH, parents, daughters_Crich, daughters_Orich,
                    daughters_2, daughters_3, grains, atoms, atoms_plus)

def get_species_filename(chemistry_type):
    return f"rate12_complex_atomic_{chemistry_type}.specs"


def get_daughters_for_chemistry(chemistry_type):
    if chemistry_type == "Orich":
        return daughters_Orich
    return daughters_Crich


def select_particle_ids(particle_IDs_file, trace_dir,
                        max_particle_id=None, start_index=2, n_select=None,
                        min_data_rows=1, quiet=False):
    """Select particle IDs with optional spacing and file-existence checks.

    When *quiet* is True, informational messages go to stderr so that stdout
    contains only the data (used by --count mode).
    """
    import sys
    def _info(msg):
        if quiet:
            print(msg, file=sys.stderr)
        else:
            print(msg)
    def _has_enough_data_rows(phys_file):
        data_rows = 0
        with open(phys_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                data_rows += 1
                if data_rows > min_data_rows:
                    return True
        return False

    with open(particle_IDs_file, 'r') as f:
        particle_IDs = [int(line.strip()) for line in f if line.strip()]

    if max_particle_id is not None:
        particle_IDs = [pid for pid in particle_IDs if pid <= max_particle_id]
    particle_IDs = [pid for pid in particle_IDs if (trace_dir / f"{pid}.phys").exists()]
    particle_IDs = [
        pid for pid in particle_IDs
        if _has_enough_data_rows(trace_dir / f"{pid}.phys")
    ]
    _info(f"Found {len(particle_IDs)} valid particle IDs in {trace_dir}" + (f" (max ID={max_particle_id})" if max_particle_id is not None else ""))

    if n_select is None:
        selected = particle_IDs[start_index:]
    else:
        n_pick = min(n_select, len(particle_IDs))
        indices = np.linspace(start_index, len(particle_IDs) - 1, num=n_pick, dtype=int)
        selected = [particle_IDs[i] for i in indices]

    _info(f"Returning {len(selected)} particle IDs (after start_index={start_index} offset)")
    return selected


def convert_phys(in_file, out_file) -> None:
    """Read a .phys trace file and write the chemistry model input file.

    Input columns:
        time(s)  x(AU)  y(AU)  z(AU)  n(cm-3)  T(K)  A_UV(mag)  A_V(mag)

    Output columns:
        XCOORD(pc)  YCOORD(pc)  ZCOORD(pc)  DENS(cm-3)
        TEMPGAS(K)  TEMPDUST(K)  AV(mag)  ZETAXR  ZETACR  TIME(s)
    """
    rows = []
    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = line.split()
            rows.append({
                'time_str': values[0],
                'x': float(values[1]),
                'y': float(values[2]),
                'z': float(values[3]),
                'dens_str': values[4],
                'temp_str': values[5],
                'a_uv_str': values[6],
                'av_str': values[7],
            })

    with open(out_file, 'w') as f:
        for row in rows:
            x_pc = (row['x'] * u.AU).to(u.pc).value
            y_pc = (row['y'] * u.AU).to(u.pc).value
            z_pc = (row['z'] * u.AU).to(u.pc).value
            # Format: X(pc) Y(pc) Z(pc) DENS TEMPGAS TEMPDUST AV ZETAXR ZETACR TIME
            f.write(
                f"    {x_pc:.14e}"
                f"    {y_pc:.14e}"
                f"    {z_pc:.14e}"
                f"    {row['dens_str']}"
                f"    {row['temp_str']}"
                f"    {row['temp_str']}"   # TEMPDUST = TEMPGAS
                f"    {row['av_str']}"
                f"    0.00000e+00"          # ZETAXR
                f"    0.00000e+00"          # ZETACR
                f"    {row['time_str']}\n"
            )


def convert_csphyspar(input_file, output_dir, velocity_km_s=15.0, n_slices=None):
    """Convert csphyspar_smooth.out to evolving-model input format.

    Parameters
    ----------
    input_file : Path
    output_dir : Path
    velocity_km_s : float
    n_slices : int or None
        If None, write a single file; otherwise write *n_slices* files, each
        starting at an equal index offset into the data.
    """
    velocity_cm_s = velocity_km_s * 1e5

    data_lines = [line.split() for line in open(input_file)
                  if line.strip() and not line.strip().startswith(('**', 'RADIUS'))]
    data_array = np.array([[float(p) for p in line[:4]] for line in data_lines])

    radius_cm, n_h2, temp, av = (data_array[:, 0], data_array[:, 1],
                                  data_array[:, 2], data_array[:, 3])
    xcoord_pc = (radius_cm * u.cm).to(u.pc).value
    time_s = radius_cm / velocity_cm_s
    zeros = np.zeros_like(xcoord_pc)

    print(f"Read {len(radius_cm)} points from {input_file.name}")
    print(f"Radius: {xcoord_pc.min():.3e} - {xcoord_pc.max():.3e} pc")
    print(f"Time: {time_s.min():.3e} - {time_s.max():.3e} s")

    def _write(data_dict, filepath):
        with open(filepath, 'w') as f:
            for i in range(len(data_dict['X'])):
                f.write(
                    f"{data_dict['X'][i]:24.17e}    {data_dict['Y'][i]:24.17e}    "
                    f"{data_dict['Z'][i]:24.17e}    {data_dict['DENS'][i]:24.17e}    "
                    f"{data_dict['TGAS'][i]:24.17e}    {data_dict['TDUST'][i]:24.17e}    "
                    f"{data_dict['AV'][i]:24.17e}    {data_dict['ZXR']:.5e}    "
                    f"{data_dict['ZCR']:.5e}    {data_dict['TIME'][i]:24.17e}\n"
                )

    if n_slices is None:
        output_file = output_dir / 'csphyspar_converted.txt'
        _write({'X': xcoord_pc, 'Y': zeros, 'Z': zeros, 'DENS': n_h2,
                'TGAS': temp, 'TDUST': temp, 'AV': av,
                'ZXR': 0.0, 'ZCR': 0.0, 'TIME': time_s}, output_file)
        print(f"Wrote {output_file}")
    else:
        n_points = len(xcoord_pc)
        step_size = n_points // n_slices
        print(f"\nCreating {n_slices} slices with ~{step_size} point offset:")
        for i in range(n_slices):
            start_idx = i * step_size
            if start_idx >= n_points:
                break
            output_file = output_dir / f'particle_slice_{i}.txt'
            _write({'X': xcoord_pc[start_idx:], 'Y': zeros[start_idx:],
                    'Z': zeros[start_idx:], 'DENS': n_h2[start_idx:],
                    'TGAS': temp[start_idx:], 'TDUST': temp[start_idx:],
                    'AV': av[start_idx:], 'ZXR': 0.0, 'ZCR': 0.0,
                    'TIME': time_s[start_idx:]}, output_file)
            print(f"Slice {i}: r ≥ {xcoord_pc[start_idx]:.3e} pc, "
                  f"{n_points - start_idx:3d} points -> {output_file.name}")


def write_analysis_params(particle_ids, output_file):
    """Write a template analysis_params.txt pre-filled with all particle IDs.

    All entries default to ANA = F.  Edit manually: set the second column to T
    for any particle you want analysed, and set the third column to the radius
    (in cm) at which to run the analyse subroutine.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    col_w = max(len(str(pid)) for pid in particle_ids)
    with open(output_file, 'w') as f:
        f.write(f"# {'particle_id':<{col_w}}  analyse(T/F)  r_coord_cm\n")
        f.write(f"# {'----------':<{col_w}}  ------------  ----------\n")
        for pid in particle_ids:
            f.write(f"  {pid:<{col_w}}  {'F':<12}  0\n")
    print(f"Created analysis params template: {output_file}")


def load_analysis_params(analysis_params_file):
    """Load per-particle analysis settings.

    Returns
    -------
    dict
        ``{particle_id: (ana: bool, r_coord_cm: float)}``
    """
    params = {}
    with open(analysis_params_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            pid = int(parts[0])
            ana = parts[1].strip().upper().startswith('T')
            r_coord_cm = float(parts[2])
            params[pid] = (ana, r_coord_cm)
    return params


def find_grid_point(model_base, particle_id, r_coord_cm):
    """Return the 1-based grid index whose radius is closest to *r_coord_cm*.

    Reads the ev_output file when available; falls back to the input trace file.
    """
    ev_file = model_base / 'ev_output' / f'ev_{particle_id}.dat'

    if ev_file.exists():
        with open(ev_file, 'r') as f:
            raw_lines = f.readlines()
        col_names = raw_lines[4].split()
        ri = col_names.index('R')
        r_values = []
        for line in raw_lines[6:]:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            r_values.append(float(stripped.split()[ri]))
    else:
        pc_to_cm = (1.0 * u.pc).to(u.cm).value
        input_file = model_base / 'input' / f'{particle_id}.txt'
        r_values = []
        with open(input_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = stripped.split()
                x = float(parts[0]) * pc_to_cm
                y = float(parts[1]) * pc_to_cm
                z = float(parts[2]) * pc_to_cm
                r_values.append(np.sqrt(x**2 + y**2 + z**2))

    r_arr = np.array(r_values)
    idx = int(np.argmin(np.abs(r_arr - r_coord_cm)))
    return idx + 1  # 1-based


def write_file_params(model_root, model_base, particle_id,
                      chemistry_type='Crich', phantom=True, ana=False, iana=0,
                      output_subdir='output'):
    """Write file_parameters_<particle_id>.txt for the Fortran model."""
    species_filename = get_species_filename(chemistry_type)
    rel_prefix = model_base.relative_to(model_root).as_posix()
    file_params_path = (model_base / 'param' / 'trace_file_param'
                        / f"file_parameters_{particle_id}.txt")
    with open(file_params_path, "w") as f:
        f.write("! FILE PARAMETERS FOR ENVELOPE MODEL\n")
        f.write("! Physical conditions input file\n")
        if phantom:
            f.write(f"{rel_prefix}/input/{particle_id}.txt\n")
        else:
            f.write(f"{rel_prefix}/input/particle_slice_{particle_id}.txt\n")
        f.write("! Reaction file\n")
        f.write("./rate12_complex.rates\n")
        f.write("! Species file\n")
        f.write(f"./{species_filename}\n")
        f.write("! Binding energies file\n")
        f.write("./rate12_binding.dat\n")
        f.write("! Reaction parameters file\n")
        f.write("param/reaction_parameters.txt\n")
        f.write("! Grain parameters file\n")
        f.write("param/grain_parameters.txt\n")
        f.write("! Radiation field parameters file\n")
        f.write("param/radiation_parameters.txt\n")
        f.write("! Output abundances file\n")
        f.write(f"{rel_prefix}/{output_subdir}/model_output_{particle_id}.dat\n")
        f.write("! Output rates file\n")
        f.write(f"{rel_prefix}/{output_subdir}/model_rates_{particle_id}.dat\n")
        f.write("! Call analyse subroutine (T(RUE) or F(ALSE))?\n")
        f.write("T\n" if ana else "F\n")
        f.write("! Grid point to run analyse\n")
        f.write(f"{iana}\n")
        f.write("! Analyse file\n")
        f.write(f"{rel_prefix}/{output_subdir}/model_analyse_{particle_id}.dat\n")
    return file_params_path


def postprocess_ev_output(filepath, to_cm=True):
    """Insert R column after Z in an ev_output file, optionally converting X/Y/Z to cm."""
    filepath = Path(filepath)
    if not filepath.exists():
        return

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()

    # File layout from evolve_output.pl:
    #   line 0: '# Collapse Model Output'
    #   line 1: blank
    #   line 2: '# ev_output/ev_<id>.dat ...'
    #   line 3: blank
    #   line 4: column names  (X  Y  Z  DENSITY ...)
    #   line 5: '#(PC) (PC) ...' units comment
    #   lines 6+: data rows
    NAMES_IDX = 4
    UNITS_IDX = 5

    col_names = raw_lines[NAMES_IDX].split()
    xi = col_names.index('X')
    yi = col_names.index('Y')
    zi = col_names.index('Z')

    pc_to_cm = (1.0 * u.pc).to(u.cm).value
    unit_str = '(CM)' if to_cm else '(PC)'

    new_col_names = col_names[:zi + 1] + ['R'] + col_names[zi + 1:]
    names_line = '  '.join(f'{n:<16}' for n in new_col_names) + '\n'

    if UNITS_IDX < len(raw_lines) and raw_lines[UNITS_IDX].lstrip().startswith('#'):
        old_units = raw_lines[UNITS_IDX].lstrip('#').split()
        new_units = old_units[:zi + 1] + [unit_str] + old_units[zi + 1:]
        if to_cm:
            new_units[xi] = unit_str
            new_units[yi] = unit_str
            new_units[zi] = unit_str
        units_line = '#  ' + '  '.join(f'{u:<16}' for u in new_units) + '\n'
        data_start = UNITS_IDX + 1
    else:
        units_line = raw_lines[UNITS_IDX]
        data_start = UNITS_IDX

    new_lines = raw_lines[:NAMES_IDX] + [names_line, units_line]

    for line in raw_lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
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
            parts[xi] = f'{x:.5e}'
            parts[yi] = f'{y:.5e}'
            parts[zi] = f'{z:.5e}'
        r = np.sqrt(x**2 + y**2 + z**2)
        parts = parts[:zi + 1] + [f'{r:.5e}'] + parts[zi + 1:]
        new_lines.append('  '.join(parts) + '\n')

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def run_model(model_root, file_params_path):
    rel_file_params = file_params_path.relative_to(model_root).as_posix()
    os.system(f"cd {model_root} && time ./model {rel_file_params}")


def convert_outmodel_to_evolve_output(model_root, model_base, particle_id,
                                      chemistry_type='Crich', to_cm=True):
    daughters = get_daughters_for_chemistry(chemistry_type)
    molecules = parents + daughters + daughters_2 + daughters_3 + atoms + atoms_plus
    molecule_str = " ".join(molecules)
    rel_prefix = model_base.relative_to(model_root).as_posix()
    os.system(
        f"cd {model_root} && "
        f"perl evolve_output.pl {rel_prefix}/output/model_output_{particle_id}.dat "
        f"{rel_prefix}/ev_output/ev_{particle_id} {molecule_str}"
    )
    ev_file = model_base / 'ev_output' / f'ev_{particle_id}.dat'
    postprocess_ev_output(ev_file, to_cm=to_cm)


def setup_directories(base_path, particle_ids=None, analysis=False,
                      analysis_params_path=None, interactive=True):
    """Create / clear working directories.

    Parameters
    ----------
    base_path : Path
    particle_ids : list of int, optional
        Used to populate the analysis params template when *analysis* is True.
    analysis : bool
        When True, ensure the analysis parameters file exists (or create a
        template and stop so the user can edit it).
    analysis_params_path : Path, optional
        Explicit path for analysis parameters. If omitted, defaults to
        ``base_path / 'param' / 'analysis_params.txt'`` for backward compatibility.
    interactive : bool
        When True (local mode), prompt before emptying directories.
        When False (SLURM mode), silently create directories without prompting.

    Returns
    -------
    bool
        True if setup succeeded, False if the user chose to cancel.
    """
    # Always ensure the chemistry root exists (e.g. .../evolving_model/Crich or Orich).
    base_path.mkdir(parents=True, exist_ok=True)

    directories = ['param/trace_file_param', 'ev_output', 'input', 'output', 'analyse_output']

    for dir_name in directories:
        dir_path = base_path / dir_name

        if not interactive:
            # SLURM/parallel mode: never prompt, never wipe, always ensure layout exists.
            dir_path.mkdir(parents=True, exist_ok=True)
            continue

        if dir_path.exists():
            if any(dir_path.iterdir()):
                print(f"\nDirectory '{dir_name}' already exists and contains files.")
                response = input(f"Empty '{dir_name}' directory? [y/n]: ").strip().lower()
                if response in ('y', 'yes'):
                    print(f"   Removing all files in {dir_name}/...")
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"   ✓ {dir_name}/ cleared")
                else:
                    print(f"✗ Keeping existing files in {dir_name}/")
            else:
                print(f"✓ {dir_name}/ exists (empty)")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {dir_name}/")

    if analysis:
        if analysis_params_path is None:
            analysis_params_path = base_path / 'param' / 'analysis_params.txt'
        if not analysis_params_path.exists():
            if interactive:
                write_analysis_params(particle_ids, analysis_params_path)
                sys.exit(
                    f"\nPlease edit the analysis parameters file at "
                    f"{analysis_params_path} to enable per-particle analysis, "
                    f"then re-run the script."
                )
            else:
                raise FileNotFoundError(
                    f"Analysis params file not found: {analysis_params_path}\n"
                    "Generate a template by running locally first."
                )
        else:
            print(f"✓ Analysis params file already exists: {analysis_params_path}")

    return True


# ============================================================
#  Entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False,
        help=(
            "Run in SLURM array mode: non-interactive, processes one particle "
            "selected by SLURM_ARRAY_TASK_ID. Set automatically by the slurm script."
        ),
    )
    parser.add_argument(
        "--analysis", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Enable/disable analysis mode (--analysis / --no-analysis). "
            "Overrides the in-script default and the ANALYSIS env var. "
            "Can also be set via ANALYSIS=true|false (e.g. ANALYSIS=true sbatch ...)."
        ),
    )
    parser.add_argument(
        "--chemistry-type", choices=["Crich", "Orich"], default=None,
        help=(
            "Chemistry type to run (Crich or Orich). Overrides the in-script "
            "default and CHEMISTRY_TYPE env var."
        ),
    )
    parser.add_argument(
        "--count", action="store_true", default=False,
        help=(
            "Print the number of selected particle IDs and exit. "
            "Used by the submission wrapper to set the SLURM array size."
        ),
    )
    args, _ = parser.parse_known_args()  # parse_known_args ignores Jupyter kernel args
    return args


def main():
    args = parse_args()
    parallel = args.parallel  # True when launched by the SLURM script

    # ------------------------------------------------------------------ CONFIG
    # Edit these variables to change what the script does.
    # They apply equally in local and parallel mode.

    pmf            = 'wind_v10'
    mf             = 'evolving_model'
    chemistry_type = 'Crich'  # default for interactive use
    phantom        = True
    model_1D       = False
    to_cm          = True   # Convert X/Y/Z/R to cm in ev_output files
    analysis       = False  # default for interactive use
    # CLI flag takes highest priority, then CHEMISTRY_TYPE env var, then default above
    if args.chemistry_type is not None:
        chemistry_type = args.chemistry_type
    elif 'CHEMISTRY_TYPE' in os.environ:
        chemistry_type = os.environ['CHEMISTRY_TYPE']
        if chemistry_type not in ('Crich', 'Orich'):
            raise ValueError(
                f"Invalid CHEMISTRY_TYPE='{chemistry_type}'. Expected 'Crich' or 'Orich'."
            )
    # CLI flag takes highest priority, then ANALYSIS env var, then the default above
    if args.analysis is not None:
        analysis = args.analysis
    elif 'ANALYSIS' in os.environ:
        analysis = os.environ['ANALYSIS'].lower() in ('1', 'true', 'yes')

    # 1-D model input (only used when model_1D = True)
    model_1D_output = (f'complete_model_Mdot_Vinf_{chemistry_type}/models/'
                       f'model_2025-10-22h14-41-04/csphyspar_smooth.out')

    # Particle selection
    max_particle_id = None
    min_data_rows   = 3 
    start_index     = 2
    n_select        = None

    # ------------------------------------------------------------------ PATHS
    tracesf   = f'traces/{pmf}/trace_output_with_av'
    model_root = BASE_PATH / mf
    model_base = model_root / chemistry_type
    trace_dir  = BASE_PATH / tracesf
    particle_IDs_file = BASE_PATH / 'traces' / pmf / 'particle_IDs.txt'
    analysis_params_file = model_root / 'param' / f'analysis_params_{chemistry_type}.txt'

    # ------------------------------------------------------------------ PARTICLE IDS
    particle_IDs = select_particle_ids(
        particle_IDs_file, trace_dir,
        max_particle_id=max_particle_id,
        start_index=start_index,
        n_select=n_select,
        min_data_rows=min_data_rows,  # require at least 3 data rows in the .phys file
        quiet=args.count,
    )

    if args.count:
        print(len(particle_IDs))
        return

    # Custom overrides – adjust as needed
    # if len(particle_IDs) > 1:
    #     particle_IDs[1] = 29823
    # if len(particle_IDs) > 2:
    #     particle_IDs[2] = 46371
    # particle_IDs = [29823, 46371]  # uncomment to override list entirely

    if not particle_IDs:
        raise RuntimeError("No valid particle IDs found.")

    # ------------------------------------------------------------------ DIRECTORIES
    if parallel:
        # Non-interactive: silently create dirs, no prompts
        setup_directories(
            model_base, particle_IDs, analysis=analysis,
            analysis_params_path=analysis_params_file, interactive=False,
        )
    else:
        print("=" * 80)
        print("DIRECTORY SETUP")
        print("=" * 80)
        if not setup_directories(
            model_base, particle_IDs, analysis=analysis,
            analysis_params_path=analysis_params_file, interactive=True,
        ):
            print("\n✗ Setup cancelled by user")
            return
        print("=" * 80)
        print()

    # ------------------------------------------------------------------ ANALYSIS PARAMS
    analysis_params = None
    if analysis:
        analysis_params = load_analysis_params(analysis_params_file)

    output_subdir = 'analyse_output' if analysis else 'output'

    # ------------------------------------------------------------------ WHICH PARTICLES TO RUN
    if parallel:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_id is None:
            raise RuntimeError(
                "--parallel was set but SLURM_ARRAY_TASK_ID is not defined. "
                "Did you submit this script via sbatch --array?"
            )
        idx = int(array_id)
        if idx < 0 or idx >= len(particle_IDs):
            raise IndexError(
                f"SLURM_ARRAY_TASK_ID={array_id} is out of range "
                f"(valid: 0–{len(particle_IDs) - 1})"
            )
        target_IDs = [particle_IDs[idx]]
        print(f"SLURM task {array_id}: processing particle {target_IDs[0]}")
    else:
        target_IDs = particle_IDs

    # ------------------------------------------------------------------ 1-D MODEL
    if model_1D:
        input_file = BASE_PATH / model_1D_output
        output_folder = model_base / 'input'
        convert_csphyspar(input_file, output_folder, n_slices=5)
        for i in tqdm(range(5), total=5):
            file_params_path = write_file_params(
                model_root, model_base, i,
                chemistry_type=chemistry_type, phantom=False,
                output_subdir=output_subdir,
            )
            run_model(model_root, file_params_path)
            if not analysis:
                convert_outmodel_to_evolve_output(
                    model_root, model_base, i,
                    chemistry_type=chemistry_type, to_cm=to_cm,
                )

    # ------------------------------------------------------------------ PHANTOM PARTICLES
    if phantom:
        print(f"Running {chemistry_type} {mf} on {len(target_IDs)} phantom particle trace(s) from {pmf}")
        for particle_ID in tqdm(target_IDs, total=len(target_IDs)):
            input_file  = BASE_PATH / tracesf / f'{particle_ID}.phys'
            output_file = model_base / 'input' / f'{particle_ID}.txt'

            if not input_file.exists():
                raise FileNotFoundError(f"Missing trace file: {input_file}")

            convert_phys(input_file, output_file)

            if analysis_params is not None and particle_ID in analysis_params:
                ana_flag, r_coord = analysis_params[particle_ID]
                iana = find_grid_point(model_base, particle_ID, r_coord) if ana_flag else 0
                file_params_path = write_file_params(
                    model_root, model_base, particle_ID,
                    chemistry_type=chemistry_type, ana=ana_flag, iana=iana,
                    output_subdir=output_subdir,
                )
            else:
                file_params_path = write_file_params(
                    model_root, model_base, particle_ID,
                    chemistry_type=chemistry_type,
                    output_subdir=output_subdir,
                )

            run_model(model_root, file_params_path)
            if not analysis:
                convert_outmodel_to_evolve_output(
                    model_root, model_base, particle_ID,
                    chemistry_type=chemistry_type, to_cm=to_cm,
                )


if __name__ == '__main__':
    main()
