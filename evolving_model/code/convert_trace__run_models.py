import numpy as np
import os
from pathlib import Path
from astropy import units as u
from tqdm import tqdm

def convert_phys(in_file, out_file) -> None:
    """
    - Read a .phys file preserving exact string representations.
    - Convert to chemistry model input format, convert coordinates from AU to pc.
    - Write output file
    
    Input columns:
    time(s)   x(AU)   Y(AU)   Z(AU)   n(cm-3)   T(K)   A_UV(mag)

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
                'av_str': values[6],
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
    data_array = np.array([[float(p) for p in line[:10]] for line in data_lines if len(line) >= 10]) # Convert to array
    
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
            
            output_file = output_dir / f'particle_slice_{i+1:02d}.txt'
            data_dict = {'X': xcoord_pc[start_idx:], 'Y': zeros[start_idx:], 'Z': zeros[start_idx:],
                        'DENS': n_h2[start_idx:], 'TGAS': temp[start_idx:], 'TDUST': temp[start_idx:],
                        'AV': av[start_idx:], 'ZXR': 0.0, 'ZCR': 0.0, 'TIME': time_s[start_idx:]}
            write_output(data_dict, output_file)
            print(f"Slice {i+1:2d}: r ≥ {xcoord_pc[start_idx]:.3e} pc, {n_points - start_idx:3d} points -> {output_file.name}")

def write_file_params(particle_ID):
    fpfile = open(savedirmain / mf / 'param' / 'trace_file_param' /
                      f"file_parameters_{particle_ID}.txt", "w")
    fpfile.write("! FILE PARAMETERS FOR ENVELOPE MODEL\n")
    fpfile.write("! Physical conditions input file\n")      
    fpfile.write(f"input/{particle_ID}.txt\n")
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

def run_model(particle_ID):
    os.system(f"cd {savedirmain / mf} && "
            f"time ./model param/trace_file_param/file_parameters_{particle_ID}.txt")
    
def convert_outmodel_to_evolve_output(particle_ID):
    parents = ["He", "CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN", 
       "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS", "Mg", "Na", "Fe"]
    daughters = ['C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N'] # Crich 
    molecules = parents + daughters
    molecule_str = " ".join(molecules) # Single space separated string
    
    os.system(f"cd {savedirmain / mf} && "
        f"perl evolve_output.pl output/model_output_{particle_ID}.dat ev_output/ev_{particle_ID} {molecule_str}")
    
def main():
    global savedirmain, tracesf, mf
    savedirmain = Path('/Users/sam/Documents/GitHub/Chemistry')
    tracesf = 'Phantom trace/trace_output'
    mf = 'evolving_model'
    model_1D_output = 'complete_model_Mdot_Vinf_Crich/models/model_2025-10-22h14-41-04/csphyspar_smooth.out'

    particle_IDs = [
    5001,
    13775,
    22549,
    31323,
    40097,
    48871,
    57645,
    66419,
    75193,
    83967,
    92741,
    101515,
    110289,
    119063,
    127837
    ]

    input_file = savedirmain / model_1D_output
    output_folder = savedirmain / mf / 'input'
    convert_csphyspar(input_file, output_folder, n_slices = 5)

    for particle_ID in tqdm(particle_IDs, total=len(particle_IDs)):
        input_file = savedirmain / tracesf / f'{particle_ID}.phys'
        output_file = savedirmain / mf / 'input' / f'{particle_ID}.txt'
        # convert_phys(input_file, output_file)
        # print(f"Converted {particle_ID}.phys to {particle_ID}.txt")

        # write_file_params(particle_ID)
        # print(f"Wrote file parameters for particle {particle_ID}")

        # run_model(particle_ID)

        # convert_outmodel_to_evolve_output(particle_ID)

    # print("DONE!")
    # print(f"Output columns:")
    # print("  XCOORD(pc), YCOORD(pc), ZCOORD(pc), DENS(cm^-3),")
    # print("  TEMPGAS(K), TEMPDUST(K), AV(mag), ZETAXR, ZETACR, TIME(s)")

if __name__ == '__main__':
    main()
