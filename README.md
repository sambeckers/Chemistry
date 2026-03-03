
# 3D chemical modelling of AGB outflows
Python + Fortran pipeline for running chemistry models on SPH particle traces from AGB stellar wind simulations (Phantom), and comparing them to 1D models via KoekenBak.

---

## Repository structure

```
Chemistry/
├── config.py                  # ← NOT tracked by git — you must create this
├── molecules.py               # Shared molecule lists
├── evolving_model/            # Fortran chemistry model source + reaction files
│   ├── Crich/                 # Working directories for C-rich runs
│   └── Orich/                 # Working directories for O-rich runs
├── KoekenBak/kb/              # Local KoekenBak Python package (1D model)
├── traces/                    # ← NOT tracked by git — particle trace files go here
├── output_1D/                 # 1D model output (KoekenBak)
├── figures/                   # Saved plots
└── code/
    ├── run_evolving_models.py          # Main driver: run chemistry on Phantom traces
    ├── run_em_parallel.slurm           # SLURM array job submission script
    ├── run_plot_1D_model.py            # Plot all 1D model results
    ├── run_plot_single_config_1D_model.py  # Plot a single Mdot/Vinf configuration
    └── plotting/
        ├── plot_avg_abundance_radius.py    # Average 3D abundance vs radius
        ├── plot_evolution_all.py           # Per-particle evolution traces
        ├── plot_evolution_single.py        # Single-particle deep-dive
        ├── plot_auv_av_wind_v10.py         # AUV/AV diagnostic plot
        ├── inspect_particles.py            # Interactive HTML particle explorer
        └── plot_utils.py                   # Shared axis-limit & tick helpers
```

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| `gfortran` ≥ 9 | Compile the Fortran chemistry model |
| `conda` (Miniconda / Anaconda) | Python environment management |
| LaTeX (`pdflatex` + `times` package) | Matplotlib `text.usetex` rendering in plots |

Install `gfortran` on macOS via Homebrew:
```bash
brew install gcc
```

On Linux (Debian/Ubuntu):
```bash
sudo apt install gfortran
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Chemistry.git
cd Chemistry
```

### 2. Create a conda environment

....
> The plotting scripts also require `matplotlib`'s `text.usetex=True` renderer. Make sure a working LaTeX installation (with `times.sty`) is on your `PATH`.

### 3. Create `config.py`

`config.py` is excluded from version control (it is listed in `.gitignore`) because it contains a machine-specific path. Create it at the repo root:

```python
# config.py
from pathlib import Path
from molecules import (
    atoms, atoms_plus,
    daughters_2, daughters_3,
    daughters_Crich, daughters_Crich_no_CN, daughters_Orich,
    grains, molecules_plot,
    parents, parents_He_extended,
    pd_daughters, pd_parents,
)

# ← Change this to the absolute path where you cloned the repo
BASE_PATH = Path('/path/to/your/Chemistry')
```
---

## Step 1 — Compile the Fortran chemistry model

The evolving model is written in Fortran 77/90. Compile it once before running:

```bash
cd evolving_model
make
```

This runs:
```
gfortran -O2 -w -std=legacy -o model \
    inputmodel.f main.f rate12_complex_odes.f drive.f dvode.f subs.f shielding.f ismana.f
```

A compiled binary called `model` will appear in `evolving_model/`. The model binary is not tracked by git; you must recompile on each machine.

> **Intel Fortran (`ifort`)**: an alternative `ifort` command is commented out at the top of the `makefile` — uncomment it if ifort is available and preferred.

---

## Step 2 — Provide particle trace files

The `traces/` directory is not tracked by git. Each trace is a `.phys` file with columns:

```
time(s)  x(AU)  y(AU)  z(AU)  n(cm⁻³)  T(K)  A_UV(mag)  A_V(mag)
```

Expected layout:

```
traces/
└── wind_v10/
    ├── particle_IDs.txt          # one integer particle ID per line
    └── trace_output_with_av/
        ├── 12345.phys
        ├── 29823.phys
        └── ...
```

The name of the subdirectory (`wind_v10`) is set by the `pmf` variable in `run_evolving_models.py`.

---

## Step 3 — Run chemistry models

### Local (interactive)

Edit the **CONFIG section** near the top of `main()` in `code/run_evolving_models.py`:

| Variable | Description |
|---|---|
| `pmf` | Trace folder name inside `traces/` (e.g. `'wind_v10'`) |
| `chemistry_type` | `'Crich'` or `'Orich'` |
| `n_select` | Number of particles to sample |
| `analysis` | `True` to run the Fortran analyse subroutine |

Then run:

```bash
cd code
python run_evolving_models.py
```

The script will interactively ask whether to empty existing output directories before proceeding.

### HPC (SLURM array)

Edit the paths in `run_em_parallel.slurm` (particularly `WORKDIR` and the `conda activate` path), then submit:

```bash
# C-rich, 15 particles (array index 0–14)
sbatch run_em_parallel.slurm Crich

# O-rich with analysis
sbatch run_em_parallel.slurm Orich analysis
```

Each array task processes one particle independently. `n_select` in the script must match `--array=0-<n_select-1>` in the SLURM header.

### Output

Results appear in `evolving_model/Crich/` (or `Orich/`):

```
evolving_model/Crich/
├── input/          # converted trace files (.txt) fed to Fortran
├── output/         # raw Fortran output (model_output_<id>.dat, model_rates_<id>.dat)
├── ev_output/      # post-processed evolution files (ev_<id>.dat)
├── param/
│   └── trace_file_param/   # per-particle file_parameters files
└── analyse_output/ # (only when analysis=True)
```

---

## Step 4 — Run the 1D model (optional)

The 1D model uses KoekenBak with pre-built input grids in `KoekenBak/input/`.

```bash
cd code
python run_plot_single_config_1D_model.py   # single Mdot/Vinf configuration
python run_plot_1D_model.py                 # all configurations
```

Set `CRICH = True/False` near the top of either script to switch chemistry. Output is saved to `output_1D/`.

---

## Step 5 — Plotting

All plotting scripts live in `code/plotting/` and read `BASE_PATH` from `config.py`.

| Script | What it produces |
|---|---|
| `plot_evolution_all.py` | Per-particle density/temperature/AV/abundance traces |
| `plot_auv_av_wind_v10.py` | AUV / AV diagnostic for the wind |
| `inspect_particles.py` | Interactive HTML explorer (opens in browser) |
| `plot_ab_mean.py` | Average 3D abundance vs radius; optional 1D comparison |

Run from the `code/plotting/` directory:

```bash
cd code/plotting
python plot_avg_abundance_radius.py
python inspect_particles.py   # writes HTML to figures/Inspection/{Crich,Orich}/index.html
```

Figures are written to the `figures/` subdirectory.

---

## Chemistry types

| Type | Parent inputs | Daughter species tracked |
|---|---|---|
| `Crich` | CO, N₂, CH₄, NH₃, H₂S, HCP, H₂O, C₂H₂, HCN, CS, SiC₂, HCl, HF, C₂H₄, SiO, SiS | CN, C₂H, C₄H, C₆H, HC₃N, HC₅N, HC₇N |
| `Orich` | (same parent set) | SiN, SiC, OH, CN, SiOH⁺ |

Reaction networks are in `evolving_model/rate12_complex_atomic_{Crich,Orich}.specs` and `evolving_model/rate12_complex.rates`.

---

## Notes

- `config.py` and `traces/` are in `.gitignore` — you must supply both.
- The compiled `model` binary and all `.o`/`.mod` build artifacts are also gitignored.
- Model output directories (`evolving_model/Crich/output`, etc.) are gitignored; they are created automatically by `run_evolving_models.py`.
- On macOS, `gfortran` from Homebrew (`gcc`) works. The makefile includes an ARM-compatible flag (`-march=armv8-a`) in a commented alternative if needed on Apple Silicon.
