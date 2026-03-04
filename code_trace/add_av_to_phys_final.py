#!/usr/bin/env python3
"""
Add A_V column to .phys trace files.

This script adds visual extinction (A_V) measurements as a new column to .phys 
trace files. It matches timestamps between the .phys files and the A_V data files,
handling cases where particles start tracing at different times.

Input:
  - .phys files in trace_output/
  - AV_X_nrays192_nside8.txt files in PhotoData/
  
Output:
  - Modified .phys files with A_V column in trace_output_with_av/

Usage:
  # Test with single particle
  python add_av_to_phys_final.py

  # Process all particles  
  python add_av_to_phys_final.py --all

  # Process specific particle
  python add_av_to_phys_final.py --particle 16570
"""

import os
import glob
import numpy as np
import argparse

from tqdm import tqdm

def extract_dump_number(filename):
    """Extract dump number from filename like 'AV_123_nrays192_nside8.txt'"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    return int(parts[1])

def read_av_from_file(filepath, particle_id):
    """
    Read A_V value for a specific particle from an AV timestep file.
    
    Returns:
        av_value (str): A_V value for the particle
    """
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if int(parts[0]) == particle_id:
                return parts[1]

def add_av_column(phys_file, particle_id, photodata_dir, output_file):
    """
    Add A_V column to a .phys file by exact time matching with A_V data.
    
    Args:
        phys_file: Path to input .phys file
        particle_id: Particle ID
        photodata_dir: Directory containing AV_*.txt files
        output_file: Path to output .phys file with A_V column
    """
    # Read .phys file
    with open(phys_file, 'r') as f:
        lines = f.readlines()
    
    # Find header line
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            header_idx = i
            break
    
    # Get first .phys time
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 1:
            first_phys_time = float(parts[0])
            break
    
    # Get all AV files in numeric order
    av_files = sorted(glob.glob(os.path.join(photodata_dir, 'AV_*_nrays192_nside8.txt')),
                     key=extract_dump_number)
    
    # Find the starting AV file with exact time match
    start_idx = None
    for idx, av_file in enumerate(av_files):
        with open(av_file, 'r') as f:
            for line in f:
                if line.startswith('# Time(s):'):
                    file_time = float(line.split(':')[1].strip())
                    # print(f"Checking AV file {av_file} with time {file_time:.8E}")
                    
                    if file_time == first_phys_time:
                        start_idx = idx
                        # print(start_idx)
                        break
                    break
        if start_idx is not None:
            break
    
    # Read A_V values sequentially from starting file
    av_values = []
    for av_file in av_files[start_idx:]:
        # print(av_file)
        av_value = read_av_from_file(av_file, particle_id)
        av_values.append(av_value)
    
    # Write output file
    output_lines = []
    header = lines[header_idx].rstrip()
    output_lines.append(header + "   A_V\n")
    
    # Add A_V column to each data line
    av_idx = 0
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        
        output_line = line.rstrip() + f"  {av_values[av_idx]}\n"
        output_lines.append(output_line)
        av_idx += 1
    
    # Write output
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

def process_particle(particle_id, trace_dir, photodata_dir, output_dir):
    """Process a single particle."""
    phys_file = os.path.join(trace_dir, f'{particle_id}.phys')
    
    if not os.path.exists(phys_file):
        print(f"  ✗ File not found: {phys_file}")
        return False
    
    output_file = os.path.join(output_dir, f'{particle_id}.phys')
    add_av_column(phys_file, particle_id, photodata_dir, output_file)
    
    # print(f"  ✓ {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Add A_V column to .phys trace files')
    parser.add_argument('--all', action='store_true', help='Process all particles')
    parser.add_argument('--particle', type=int, help='Process specific particle ID')
    args = parser.parse_args()
    
    # Configuration
    trace_dir = '/net/vdesk/data2/beckers/MRP/v10_a09/trace_output'
    photodata_dir = '/net/vdesk/data2/beckers/MRP/cd/PhotoData'
    output_dir = '/net/vdesk/data2/beckers/MRP/v10_a09/trace_output_with_av'
    particle_ids_file = '/net/vdesk/data2/beckers/MRP/v10_a09/particle_IDs.txt'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which particles to process
    if args.particle:
        particle_ids = [args.particle]
    elif args.all:
        with open(particle_ids_file, 'r') as f:
            particle_ids = [int(line.strip()) for line in f if line.strip()]
    else:
        # Default: test with particle 16570
        particle_ids = [5001]
        print("\n" + "="*70)
        print("TEST MODE: Processing particle 16570")
        print("Use --all to process all particles or --particle ID for specific one")
        print("="*70)
    
    print(f"\nInput trace files: {trace_dir}")
    print(f"Input A_V data: {photodata_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    success_count = 0
    for particle_id in tqdm(particle_ids, total=len(particle_ids), desc="Processing particles"):
        if process_particle(particle_id, trace_dir, photodata_dir, output_dir):
            success_count += 1
        else:
            print(f"✗ Failed to process particle {particle_id}")
    
    # Summary
    print(f"\n✓ Processed {success_count}/{len(particle_ids)} particles")

if __name__ == '__main__':
    main()
