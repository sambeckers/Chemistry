#!/usr/bin/env python3
"""
Script to reorganize A_V data from per-timestep files to per-particle files.
Memory-efficient streaming version.

Input: PhotoData folder with AV_X_nrays192_nside8.txt files (one per timestep)
Output: AV_particleID.txt files with columns: time(s), A_V(mag)
"""

import os
import glob

def extract_dump_number(filename):
    """Extract dump number from filename like 'AV_123_nrays192_nside8.txt'"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    return int(parts[1])

def reorganize_av_data_streaming(photodata_dir, particle_ids, output_dir, test_one_particle=False):
    """
    Reorganize A_V data from per-timestep to per-particle files using streaming I/O.
    Opens all output files at once and writes to them as we read input files.
    
    Args:
        photodata_dir: Directory containing AV_*.txt files
        particle_ids: List of particle IDs to process
        output_dir: Directory to save output files
        test_one_particle: If True, only process the first particle ID
    """
    # Find all AV files
    av_files = sorted(glob.glob(os.path.join(photodata_dir, 'AV_*_nrays192_nside8.txt')),
                     key=extract_dump_number)
    
    print(f"Found {len(av_files)} A_V files")
    
    # If testing, only process first particle
    if test_one_particle and len(particle_ids) > 0:
        particle_ids = [1635001]
        print(f"Testing with particle ID: {particle_ids[0]}")
    
    # Convert to set for faster lookup
    particle_ids_set = set(particle_ids)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open all output files at once
    output_files = {}
    try:
        print(f"Opening {len(particle_ids)} output files...")
        for particle_id in particle_ids:
            output_path = os.path.join(output_dir, f'AV_{particle_id}.txt')
            f = open(output_path, 'w')
            f.write(f"# Particle ID: {particle_id}\n")
            f.write(f"# Time(s)  A_V(mag)\n")
            output_files[particle_id] = f
        
        # Process each timestep file
        print("Reading timestep files and writing data...")
        for i, av_file in enumerate(av_files):
            if i % 100 == 0:
                print(f"  Processing file {i+1}/{len(av_files)}", flush=True)
            
            time = None
            
            # Read file line by line
            with open(av_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Extract time from header
                    if line.startswith('# Time(s):'):
                        time = float(line.split(':')[1].strip())
                        continue
                    
                    # Skip other comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse data line
                    parts = line.split()
                    if len(parts) == 2:
                        particle_id = int(parts[0])
                        
                        # If this is one of our particles, write to its file
                        if particle_id in particle_ids_set:
                            av_value = float(parts[1])
                            output_files[particle_id].write(f"{time:.10e}  {av_value:.10e}\n")
    
    finally:
        # Close all output files
        print("\nClosing output files...")
        for particle_id, f in output_files.items():
            f.close()
        
        print(f"✓ Successfully processed {len(output_files)} particle file(s)")
        print(f"Output directory: {output_dir}")

def main():
    # Paths
    photodata_dir = '/net/vdesk/data2/beckers/MRP/cd/PhotoData'
    particle_ids_file = '/net/vdesk/data2/beckers/MRP/v10_a09/particle_IDs.txt'
    output_dir = '/net/vdesk/data2/beckers/MRP/v10_a09/AV_per_particle'
    
    # Read particle IDs
    print(f"Reading particle IDs from {particle_ids_file}")
    with open(particle_ids_file, 'r') as f:
        particle_ids = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Found {len(particle_ids)} particle IDs")
    print(f"First few IDs: {particle_ids[:5]}\n")
    
    # Run reorganization
    # Set test_one_particle=True to test with just the first particle
    # Set test_one_particle=False to process all particles
    TEST_MODE = True  # Change to False to process all particles
    
    reorganize_av_data_streaming(photodata_dir, particle_ids, output_dir, test_one_particle=TEST_MODE)
    
    if TEST_MODE:
        print(f"\n⚠ TEST MODE: Only processed 1 particle")
        print(f"To process all {len(particle_ids)} particles, set TEST_MODE=False in the script")

if __name__ == '__main__':
    main()
