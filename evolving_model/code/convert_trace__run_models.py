import numpy as np
import os
from pathlib import Path
from astropy import units as u
from tqdm import tqdm

def convert_phys_to_chemistry_txt(in_file, out_file) -> None:
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

    for particle_ID in tqdm(particle_IDs, total=len(particle_IDs)):
        input_file = savedirmain / tracesf / f'{particle_ID}.phys'
        output_file = savedirmain / mf / 'input' / f'{particle_ID}.txt'
        convert_phys_to_chemistry_txt(input_file, output_file)
        # print(f"Converted {particle_ID}.phys to {particle_ID}.txt")

        write_file_params(particle_ID)
        # print(f"Wrote file parameters for particle {particle_ID}")

        # run_model(particle_ID)

        convert_outmodel_to_evolve_output(particle_ID)

    # print("DONE!")
    # print(f"Output columns:")
    # print("  XCOORD(pc), YCOORD(pc), ZCOORD(pc), DENS(cm^-3),")
    # print("  TEMPGAS(K), TEMPDUST(K), AV(mag), ZETAXR, ZETACR, TIME(s)")

if __name__ == '__main__':
    main()
