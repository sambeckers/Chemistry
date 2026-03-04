#!/usr/bin/env python3
"""
Fix time precision in AV files from .10e to .8e using exact simulation times.
Reads each dump file to get the precise simulation time.
"""
import glob
import re
from pathlib import Path
import plons
from tqdm import tqdm

# Configuration
map_name = 'v10_a09'
wdir = Path.cwd()
pdir = wdir.parent
photodata_dir = Path('PhotoData')

# Get all AV files and sort them numerically by dump number
av_files = glob.glob('PhotoData/AV_*_nrays192_nside8.txt')
# Sort by extracting the numeric dump number from the filename
av_files = sorted(av_files, key=lambda x: int(re.search(r'AV_(\d+)_nrays', x).group(1)))
# print(av_files[0:5])
print(f"Found {len(av_files)} AV files to process")

for filepath in tqdm(av_files, total=len(av_files), desc="Processing AV files"):
    # Extract dump number from filename: AV_1234_nrays192_nside8.txt
    filename = Path(filepath).name
    match = re.search(r'AV_(\d+)_nrays', filename)
    if not match:
        print(f"Skipping {filename}: couldn't extract dump number")
        continue
    
    dump = int(match.group(1))
    
    # Load dump file to get exact time
    dump_file = Path(pdir / f'{map_name}/wind_{dump:05d}')
    
    try:
        setupData = plons.LoadSetup(pdir, f"{map_name}/wind")
        dumpData = plons.LoadFullDump(str(dump_file), setupData)
        
        code_time = dumpData._params['time'] 
        unit_time = dumpData._params['utime']
        real_time = code_time * unit_time  # in seconds
        
        # Read the file
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Update the time header with .8e precision
        if lines[0].startswith('# Time(s):'):
            lines[0] = f'# Time(s): {real_time:.8e}\n'
            
            # Write back
            with open(filepath, 'w') as f:
                f.writelines(lines)
            
            # print(f"Fixed dump {dump}: {real_time:.8e} seconds")
        else:
            print(f"Skipping dump {dump}: unexpected header format")
            
    except Exception as e:
        print(f"Error processing dump {dump}: {e}")

print(f"\nProcessing complete!")
