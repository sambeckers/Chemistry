'''
Extract a list of particle IDs to track from a phantom simulation and write it
to a file.
'''
import os

import numpy as np
import pandas as pd
import sarracen


DIR="/Users/camille/Documents/runs/phantom/macetraining"
MODEL="v17-5"
PH_DIR=os.path.join(DIR,"3d", MODEL)
FINAL_DIR=os.path.join(DIR,"1d", MODEL)
FILENAME=MODEL

# Number of particles to analyze
PARTICLE_COUNT = 10000

# number of boundary particles to skip
SKIP_BOUNDARY = 5000

#step to account for dump files
DUMP_STEP = 1

#change units to cgs
msol_to_g = 1.9885e33  # Solar mass in grams
au_to_cm = 1.496e13  # Astronomical unit in centimeters


def get_particle_data(sdf, particle_ID) -> tuple:
    """Get the density and temperature of a particle with a given ID from a sarracen dataframe.
    Args:
        sdf (sarracen.DataFrame): The sarracen dataframe containing particle data.
        particle_ID (int): The ID of the particle to retrieve data for.
    Returns:
        tuple: A tuple containing the density and temperature of the particle.
    """
    if particle_ID not in sdf['iorig'].values:
        raise ValueError(
            f"Particle ID {particle_ID} not found in the dataframe.")
    mask = sdf['iorig'] == particle_ID
    density = sdf['rho'][mask]
    temperature = sdf['Tdust'][mask]
    x, y, z = sdf['x'][mask], sdf['y'][mask], sdf['z'][mask]
    distance = np.sqrt(x**2 + y**2 + z**2)
    return density, temperature, distance

def get_dump_file_bounds(directory, filename):
    """Get the number of the first and last dump file in a directory with a specific filename pattern.
    Args:
        directory (str): The directory to search for dump files.
        filename (str): The base filename pattern to look for.
    Returns:
        str: The number of the first and last dump file found.
    """
    #gather list of files with the right pattern
    files = [int(f.split('_')[-1]) for f in os.listdir(directory) if f.startswith(filename) and '.' not in f]
    if not files:
        raise FileNotFoundError(f"No dump files found in {directory} with the pattern {filename}.")
    # grab the first and last file numbers
    first_file = min(files)
    last_file = max(files)

    return first_file, last_file

# Get the first and last dump file
first_dump_file, last_dump_file = get_dump_file_bounds(PH_DIR, FILENAME)

#Particle IDs
sdf = sarracen.read_phantom(os.path.join(PH_DIR, f"{FILENAME}_{first_dump_file:05d}"))[0]
min_id = sdf['iorig'].min() + SKIP_BOUNDARY  # skip first 5000 particles to avoid boundary particles
sdf = sarracen.read_phantom(os.path.join(PH_DIR, f"{FILENAME}_{last_dump_file:05d}"))[0]
max_id = sdf['iorig'].max()
particle_ID = np.linspace(min_id, max_id, PARTICLE_COUNT, dtype=int)

# write particle IDs to a file
with open(os.path.join(FINAL_DIR, f"particle_IDs.txt"), 'w') as f:
    for pid in particle_ID:
        f.write(f"{pid}\n")

print(
    f"Particle IDs written to {os.path.join(FINAL_DIR, f'particle_IDs.txt')}")
