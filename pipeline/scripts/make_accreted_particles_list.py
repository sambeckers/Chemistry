#!/usr/bin/env python3
"""Write the IDs of accreted (h <= 0) particles for every Phantom dump."""
from pathlib import Path
import numpy as np
import sarracen

PHANTOM_DIR = Path("/fred/oz304/beckers/v10a09")
OUT_DIR = Path("/aphid/scratch-3month/sbeckers/v10a09_out/accreted_ids")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ADJUST: pattern of the Phantom dump files and how the number maps to dump_XXXXX.h5
files = sorted(PHANTOM_DIR.glob("wind_0*"))

for fp in files:
    num = fp.name.split("_")[-1]
    sdf, sinks = sarracen.read_phantom(str(fp), ignore_inactive=False)  # keep dead particles
    # if "iorig" not in sdf.keys():
    #     raise RuntimeError(f"{fp.name}: no particle ID column ('iorig'); columns: {list(sdf.columns)}")
    accr = sdf["iorig"].to_numpy()[sdf["h"].to_numpy() <= 0]
    np.save(OUT_DIR / f"accr_{num}.npy", np.sort(accr).astype(np.int64))
    print(f"{fp.name}: {accr.size} accreted / {len(sdf)} particles")
    
    # If the particle is accreted, print h
    for i, h in enumerate(sdf["h"].to_numpy()):
        if h <= 0:
            print(f"Particle {sdf['iorig'].to_numpy()[i]}: h = {h}")
            