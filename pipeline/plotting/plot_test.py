import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from kb import KoekenBak
from config import BASE_PATH
from code_chem.run_plot_single_config_1D_model import TARGET_MLOSS, TARGET_VELOCITY, run_model, find_star_for_config
from code_chem.plotting.n_distinct_colours import generate_colormap
DUMP_DIR   = Path("/aphid/scratch-3month/sbeckers/v10a09_out/output/dumps")
MAX_SAMPLE = 500

print(f"Reading dump files from {DUMP_DIR}...")
def read_co_mean(path):
    with h5py.File(path, "r") as f:
        n    = f.attrs["n_particles"]
        radius = f["particles/r"][0]
        ds   = f["particles/hcl"]
        if n <= MAX_SAMPLE:
            co = ds[:]
        else:
            start = np.random.randint(0, n - MAX_SAMPLE)
            co    = ds[start : start + MAX_SAMPLE]
        print(f"Read {len(co)} samples from {path.name} at radius {radius:.2f}")
    return float(radius), float(np.mean(co))

dump_files = sorted(DUMP_DIR.glob("dump_*.h5"))

with ThreadPoolExecutor(max_workers=512) as pool:
    results = list(tqdm(pool.map(read_co_mean, dump_files), total=len(dump_files)))

radii, co_means = zip(*sorted(results))

plt.figure(dpi=300)
plt.scatter(radii, co_means)
plt.xlabel("Radius [cm]")
plt.ylabel("Mean HCl abundance")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.savefig("hcl_mean.png", dpi=150)