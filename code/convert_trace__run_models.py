import numpy as np
import os
import shutil
from pathlib import Path
from astropy import units as u
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BASE_PATH, parents, daughters_Crich as daughters,
                    daughters_2, daughters_3, grains, atoms, atoms_plus)

def select_particle_ids(
    particle_IDs_file,
    trace_dir,
    max_particle_id=300000,
    start_index=2,
    n_select=15,
):
    """Select particle IDs with optional spacing and file existence checks."""
    with open(particle_IDs_file, 'r') as f:
        particle_IDs = [int(line.strip()) for line in f if line.strip()]

    particle_IDs = [pid for pid in particle_IDs if pid <= max_particle_id]
    particle_IDs = [pid for pid in particle_IDs if (trace_dir / f"{pid}.phys").exists()]
    print(f"Found {len(particle_IDs)} valid particle IDs in {trace_dir} (max ID={max_particle_id})")

    n_pick = min(n_select, len(particle_IDs))
    indices = np.linspace(start_index, len(particle_IDs)-1, num=n_pick, dtype=int)

    return [particle_IDs[i] for i in indices]

def convert_phys(in_file, out_file) -> None:
    """
    - Read a .phys file preserving exact string representations.
    - Convert to chemistry model input format, convert coordinates from AU to pc.
    - Write output file
    
    Input columns:
    time(s)   x(AU)   Y(AU)   Z(AU)   n(cm-3)   T(K)   A_UV(mag)  A_V(mag)

    Output columns: 
    XCOORD(pc), YCOORD(pc), ZCOORD(pc), DENS(cm-3),
    TEMPGAS(K), TEMPDUST(K), AV(mag), ZETAXR, ZETACR, TIME(s)
    """
    # Read .phys file
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

    # Convert to output format
    output_rows = []
    for row in rows:
        # Convert coords from AU to PC
        x_pc = (row['x'] * u.AU).to(u.pc).value
        y_pc = (row['y'] * u.AU).to(u.pc).value
        z_pc = (row['z'] * u.AU).to(u.pc).value
        
        # Keep exact string values for density, temp, av, time
        # TEMPGAS = TEMPDUST = T(K), ZETAXR = ZETACR = 0
        output_rows.append({
            'x_pc': x_pc,
            'y_pc': y_pc,
            'z_pc': z_pc,
            'dens_str': row['dens_str'],
            'temp_str': row['temp_str'],
            'av_str': row['av_str'],
            'time_str': row['time_str'],
        })

    # Write to output file
    with open(out_file, 'w') as f:
        for row in output_rows:
            # Format: X(pc), Y(pc), Z(pc), DENS, TEMPGAS, TEMPDUST, AV, ZETAXR, ZETACR, TIME
            # Coordinates are computed (new), others are exact strings from input
            line = (f"    {row['x_pc']:.14e}"
                   f"    {row['y_pc']:.14e}"
                   f"    {row['z_pc']:.14e}"
                   f"    {row['dens_str']}"
                   f"    {row['temp_str']}"
                   f"    {row['temp_str']}"  # TEMPDUST = TEMPGAS
                   f"    {row['av_str']}"
                   f"    0.00000e+00"  # ZETAXR
                   f"    0.00000e+00"  # ZETACR
                   f"    {row['time_str']}\n")
            f.write(line)

def convert_csphyspar(input_file, output_dir, velocity_km_s=15.0, n_slices=None):
    """Convert csphyspar_smooth.out to evolving model input format with optional slicing.
    
    Parameters
    ----------
    input_file : str or Path
        Path to csphyspar_smooth.out file
    output_dir : str or Path
        Directory to save output files
    velocity_km_s : float, default=15.0
        Outflow velocity in km/s
    n_slices : int or None, default=None
        Number of particle slices to create. If None, creates single file.
        Each slice starts at a different index with equal index spacing,
        ensuring each particle file has sufficient data points.
    
    Returns
    -------
    None
        Writes file(s) to output_dir
    """
    velocity_cm_s = velocity_km_s * 1e5
    
    # Read and parse data
    data_lines = [line.split() for line in open(input_file) 
                  if line.strip() and not line.strip().startswith(('**', 'RADIUS'))]
    data_array = np.array([[float(p) for p in line[:4]] for line in data_lines]) # Convert to array
    
    radius_cm, n_h2, temp, av = data_array[:, 0], data_array[:, 1], data_array[:, 2], data_array[:, 3]
    xcoord_pc = (radius_cm * u.cm).to(u.pc).value
    time_s = radius_cm / velocity_cm_s
    zeros = np.zeros_like(xcoord_pc)
    
    print(f"Read {len(radius_cm)} points from {input_file.name}")
    print(f"Radius: {xcoord_pc.min():.3e} - {xcoord_pc.max():.3e} pc")
    print(f"Time: {time_s.min():.3e} - {time_s.max():.3e} s")
    
    def write_output(data_dict, filepath):
        """Write data to file in evolving model format."""
        with open(filepath, 'w') as f:
            for i in range(len(data_dict['X'])):
                f.write(f"{data_dict['X'][i]:24.17e}    {data_dict['Y'][i]:24.17e}    "
                       f"{data_dict['Z'][i]:24.17e}    {data_dict['DENS'][i]:24.17e}    "
                       f"{data_dict['TGAS'][i]:24.17e}    {data_dict['TDUST'][i]:24.17e}    "
                       f"{data_dict['AV'][i]:24.17e}    {data_dict['ZXR']:.5e}    "
                       f"{data_dict['ZCR']:.5e}    {data_dict['TIME'][i]:24.17e}\n")
    
    if n_slices is None:
        # Create single file with all data
        output_file = output_dir / 'csphyspar_converted.txt'
        data_dict = {'X': xcoord_pc, 'Y': zeros, 'Z': zeros, 'DENS': n_h2,
                    'TGAS': temp, 'TDUST': temp, 'AV': av, 'ZXR': 0.0, 'ZCR': 0.0, 'TIME': time_s}
        write_output(data_dict, output_file)
        print(f"Wrote {output_file}")
    else:
        # Create sliced files with equal index spacing (not radial spacing)
        # This ensures each slice has reasonable amount of data
        n_points = len(xcoord_pc)
        step_size = n_points // n_slices
        
        print(f"\nCreating {n_slices} slices with ~{step_size} point offset:")
        for i in range(n_slices):
            start_idx = i * step_size
            
            if start_idx >= n_points:
                break
            
            output_file = output_dir / f'particle_slice_{i}.txt'
            data_dict = {'X': xcoord_pc[start_idx:], 'Y': zeros[start_idx:], 'Z': zeros[start_idx:],
                        'DENS': n_h2[start_idx:], 'TGAS': temp[start_idx:], 'TDUST': temp[start_idx:],
                        'AV': av[start_idx:], 'ZXR': 0.0, 'ZCR': 0.0, 'TIME': time_s[start_idx:]}
            write_output(data_dict, output_file)
            print(f"Slice {i}: r ≥ {xcoord_pc[start_idx]:.3e} pc, {n_points - start_idx:3d} points -> {output_file.name}")

def write_file_params(particle_ID):
    fpfile = open(savedirmain / mf / 'param' / 'trace_file_param' /
                      f"file_parameters_{particle_ID}.txt", "w")
    fpfile.write("! FILE PARAMETERS FOR ENVELOPE MODEL\n")
    fpfile.write("! Physical conditions input file\n")
    if phantom:      
        fpfile.write(f"input/{particle_ID}.txt\n")
    else:
        fpfile.write(f"input/particle_slice_{particle_ID}.txt\n")
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
    fpfile.write(f"output/model_output_{particle_ID}.dat\n")
    fpfile.write("! Output rates file\n")
    fpfile.write(f"output/model_rates_{particle_ID}.dat\n")
    fpfile.write("! Call analyse subroutine (T(RUE) or F(ALSE))?\n")
    fpfile.write("F\n")
    fpfile.write("! Grid point to run analyse\n")
    fpfile.write("0\n")
    fpfile.write("! Analyse file\n")
    fpfile.write(f"./model_analyse_{particle_ID}.dat\n")
    fpfile.close()

def postprocess_ev_output(filepath, to_cm=True):
    """Insert R column after Z in an ev_output file, optionally converting X/Y/Z to cm.

    evolve_output.pl writes X/Y/Z in pc.  This function computes
    R = sqrt(X^2+Y^2+Z^2) and inserts it after the Z column so downstream
    readers can use data['R'] directly without any coordinate arithmetic.
    When to_cm=True the X, Y, Z, R values are also converted to centimetres.
    The file is rewritten in-place.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()

    # File layout produced by evolve_output.pl:
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

    # New names line with R inserted after Z
    new_col_names = col_names[:zi + 1] + ['R'] + col_names[zi + 1:]
    names_line = '  '.join(f'{n:<16}' for n in new_col_names) + '\n'

    # New units comment line
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
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        parts = parts[:zi + 1] + [f'{r:.5e}'] + parts[zi + 1:]
        new_lines.append('  '.join(parts) + '\n')

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def run_model(particle_ID=None, particle_slice=None):
    os.system(f"cd {savedirmain / mf} && "
            f"time ./model param/trace_file_param/file_parameters_{particle_ID}.txt")
    
def convert_outmodel_to_evolve_output(particle_ID, to_cm=True):
    molecules = parents + daughters + daughters_2 + daughters_3 + grains + atoms + atoms_plus
    molecule_str = " ".join(molecules) # Single space separated string
    
    os.system(f"cd {savedirmain / mf} && "
        f"perl evolve_output.pl output/model_output_{particle_ID}.dat ev_output/ev_{particle_ID} {molecule_str}")
    ev_file = savedirmain / mf / 'ev_output' / f'ev_{particle_ID}.dat'
    postprocess_ev_output(ev_file, to_cm=to_cm)

def setup_directories(base_path):
    """Create or clear directories for model runs.
    
    Parameters
    ----------
    base_path : Path
        Base path for the evolving model directory
    
    Returns
    -------
    bool
        True if setup successful, False if user cancelled
    """
    directories = ['param/trace_file_param', 'ev_output', 'input', 'output']
    
    for dir_name in directories:
        dir_path = base_path / dir_name
        
        if dir_path.exists():
            # Check if directory has files
            if any(dir_path.iterdir()):
                print(f"\nDirectory '{dir_name}' already exists and contains files.")
                response = input(f"Empty '{dir_name}' directory? [y/n]: ").strip().lower()
                
                if response == 'y' or response == 'yes':
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
    
    return True
    
def main():
    global savedirmain, pmf, tracesf, mf, phantom, model_1D
    savedirmain = BASE_PATH
    pmf = 'wind_v10'
    tracesf = f'traces/{pmf}/trace_output_with_av'
    mf = 'evolving_model'
    phantom = True
    model_1D = False
    to_cm = True  # Convert X/Y/Z/R to cm in ev_output files
    model_1D_output = 'complete_model_Mdot_Vinf_Crich/models/model_2025-10-22h14-41-04/csphyspar_smooth.out'

    # Read and select particle IDs
    particle_IDs_file = savedirmain / 'traces' / pmf / 'particle_IDs.txt'
    trace_dir = savedirmain / tracesf
    particle_IDs = select_particle_ids(
        particle_IDs_file,
        trace_dir,
        max_particle_id=300000,
        start_index=2,
        n_select=15,
    )
    particle_IDs[1] = 29823
    particle_IDs[2] = 46371
    # particle_IDs = [29823, 46371]

    # Setup directories (create or clear)
    # print("=" * 80)
    # print("DIRECTORY SETUP")
    # print("=" * 80)
    # if not setup_directories(savedirmain / mf):
    #     print("\n✗ Setup cancelled by user")
    #     return
    # print("=" * 80)
    # print()

    if model_1D:
        input_file = savedirmain / model_1D_output
        output_folder = savedirmain / mf / 'input'
        convert_csphyspar(input_file, output_folder, n_slices = 5)

        for i in tqdm(range(5), total=5):
            write_file_params(i)
            run_model(i)
            convert_outmodel_to_evolve_output(i, to_cm=to_cm)

    if phantom:
        print(f"Running {mf} on {len(particle_IDs)} phantom particle traces from {pmf}")
        for particle_ID in tqdm(particle_IDs, total=len(particle_IDs)):
            input_file = savedirmain / tracesf / f'{particle_ID}.phys'
            output_file = savedirmain / mf / 'input' / f'{particle_ID}.txt'
            convert_phys(input_file, output_file)
            write_file_params(particle_ID)
            run_model(particle_ID)
            convert_outmodel_to_evolve_output(particle_ID, to_cm=to_cm)

    # print("DONE!")
    # print(f"Output columns:")
    # print("  XCOORD(pc), YCOORD(pc), ZCOORD(pc), DENS(cm^-3),")
    # print("  TEMPGAS(K), TEMPDUST(K), AV(mag), ZETAXR, ZETACR, TIME(s)")

if __name__ == '__main__':
    main()
