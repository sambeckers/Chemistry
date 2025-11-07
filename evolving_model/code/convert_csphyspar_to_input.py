"""
Convert csphyspar_smooth.out to evolving model input format and create sliced particle files.

Output format: XCOORD(PC), YCOORD(PC), ZCOORD(PC), DENS(CM^-3), TEMPGAS(K), 
               TEMPDUST(K), AV(K), ZETAXR(I), ZETACR(I), TIME(s)
"""

import numpy as np
from pathlib import Path
from astropy import units as u

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

if __name__ == '__main__':
    base_dir = Path('/Users/sam/Documents/GitHub/Chemistry')
    input_file = base_dir / 'complete_model_Mdot_Vinf_Crich/models/model_2025-10-22h14-41-04/csphyspar_smooth.out'
    output_dir = base_dir / 'evolving_model/input'
    
    # Convert without slicing
    convert_csphyspar(input_file, output_dir, velocity_km_s=15.0, n_slices=None)
    
    # Create 10 sliced particle files
    convert_csphyspar(input_file, output_dir, velocity_km_s=15.0, n_slices=5)
