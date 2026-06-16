# Chemistry

Post-processing pipeline for computing chemical abundances from Phantom SPH simulations of AGB stellar winds.

## Installation

<!-- TODO: Add installation instructions for the following -->

- **KoekenBak (KB)**: *placeholder*
- **MRP_env**: *placeholder*
- **Magritte**: *placeholder*
- **Phantom**: *placeholder*
- **Splash**: *placeholder*

## Repository layout

| Directory | Description |
|---|---|
| `pipeline/` | Batch-parallel HDF5 pipeline |
| `code_chem/` | Python wrappers around the Fortran chemistry model. `run_evolving_models.py` drives single or parallel evolving-model runs and manages input/output file handling. |
| `code_trace/` | column density raytracing. **`column_densities.py` is a dependency for the pipeline** — it computes the visual extinction ($A_V$) files the pipeline reads. |
| `evolving_model/` | Fortran source for the chemistry model (Rate12 network, DVODE solver, shielding). |
| `src-includeDust-completeSurfaceChemistry-VR-adjustEbind-IPAP/` | Rate12 input for the 1D model: reaction rates, species files, binding energies, and shielding data. |
| `plotting/`, `code_chem/plotting/`, `pipeline/plotting/` | Visualization scripts (see [Plotting](#plotting)). |

## Pipeline

The pipeline lives in `pipeline/` and runs on an HPC cluster via SLURM.
All settings are in `pipeline/config/pipeline_config.yaml`.

The recommended workflow is to run each step separately (not the end-to-end `batch.sh`, which chains everything with SLURM dependencies, which is harder to resume if a node crashes).

### Configuration

Key fields in `pipeline_config.yaml`, e.g. for v20a25:

```yaml
paths:
  phantom_dump_dir: v20a25               # Phantom dump directory (relative to base_path)
  av_dir: v20a25_out/AV                  # AV_XXXXX files from column_densities.py
  batch_output_dir: .../output/batches   # per-batch HDF5 output
  final_output_dir: .../output/dumps     # merged per-dump HDF5 output

processing:
  start_dump: 0
  end_dump: 1200
  n_batches: 10000
  n_boundary: 50000           # low-ID boundary particles to exclude
  chemistry_types: [Crich]    # or [Orich], or [Crich, Orich]
```

Paths may be absolute or relative to `paths.base_path`. See the config file for the full list.

### Pre-requisite: compute column densities

Before running the pipeline, generate the $A_V$ files with `code_trace/column_densities.py`. Can be run in parallel with a SLURM array job (submit in bash script with your configuration):

```bash
cd code_trace
./submit_cd_parallel.sh   # produces AV_XXXXX files in the configured av_dir
```

The pipeline reads these files at batch time. Without them, chemistry runs will fail.

### Step 1 — Particle discovery

Scan Phantom dumps to build a cache of particle IDs. Run once (or when the dump range changes).

```bash
cd pipeline
./workflow/submit_discover_particle_ids.sh
# Options: --serial, --max-concurrent N, --config PATH
```

This writes `particle_ids_all.npy` and `dump_time_map_cache.json`, and updates `pipeline_config.yaml` with discovered counts.

### Step 2 — Tracing

Extract physical histories (position, density, temperature, $A_V$) for each particle across all dumps using `phantomanalysis`.

```bash
./workflow/submit_tracing.sh
# Options: --after <discover_merge_job_id>, --config PATH
```

Use `--after` to chain this after the discovery merge job if needed.
Output: per-batch `tracing_batch_XXXXX.h5` files.

### Step 3 — Batch (chemistry)

Run the Fortran chemistry model on each batch of particles. Each SLURM array task loads the tracing HDF5 for its batch and produces a `batch_XXXXX.h5` file.

```bash
./workflow/submit_batch.sh
# Options: --after <tracing_job_id>, --max-concurrent N, --config PATH
```

Use `--after` to chain this after the tracing job.

### Step 4 — Scatter

Distribute and run chemistry across batches using the scatter–gather framework.

```bash
./workflow/submit_scatter_gather.sh --scatter-only
# Options: --scatter-tasks-per-job N, --resume-scatter STATUS_FILE
```

### Step 5 — Gather (merge)

Merge per-batch outputs into per-dump HDF5 files.

```bash
./workflow/submit_scatter_gather.sh --gather-only
# Options: --resume-gather STATUS_FILE
```

Final output appears in `paths.final_output_dir` as `dump_XXXXX.h5`.

### Resume mode

If batches fail, you can resubmit only the incomplete ones:

1. **Scan batch status** — run `scan_batch_status.py` to classify each batch as COMPLETED, FAILED, or RUNNING:

   ```bash
   python pipeline/scripts/scan_batch_status.py --log-root /path/to/scratch/logs --output batch_status.txt
   ```

2. **Resume** — pass the status file to `submit_batch.sh` or `submit_scatter_gather.sh`:

   ```bash
   ./workflow/submit_batch.sh --resume batch_status.txt
   # or
   ./workflow/submit_scatter_gather.sh --scatter-only --resume-scatter batch_status.txt
   ```

   Only batches not marked COMPLETED are resubmitted.

### End-to-end submission (not recommended)

`workflow/batch.sh` chains all steps (discovery → tracing → batch → scatter → merge → summary) with SLURM dependencies in a single command. This is convenient but harder to debug when individual steps fail.

```bash
./workflow/batch.sh
# Options: --max-concurrent N, --resume STATUS_FILE
```

## `code_chem`

Python wrappers for running the Fortran evolving model outside the pipeline. `run_evolving_models.py` sets up input files, calls the compiled model binary, and collects output. Useful for single-particle or small-scale runs and for running 1D reference models. Includes its own SLURM submit scripts (`submit_em_parallel.sh`).

## `code_trace`

SPH raytracing tools. Key files:

- `column_densities.py` — computes visual extinction ($A_V$) per particle per dump. **Required by the pipeline.**
- `particleIDs.py` — particle ID extraction utilities.
- `utils_raytracer.py` — raytracing helper functions.
- `submit_cd_parallel.sh` / `submit_trace_parallel.sh` — SLURM submission wrappers.

## Plotting

Visualization scripts are spread across three locations:

- **`pipeline/plotting/`** — post-pipeline analysis: slice renders (`render_slice_v2.py`), abundance point-density plots (`plot_pointdensity.py`), column-density maps (`coldens.py`), statistics (`plot_stats.py`), radial comparisons (`compare_radii.py`), and animation tools (`animate_renders_v2.py`).
- **`code_chem/plotting/`** — 1D model plots: mean abundances (`plot_ab_mean.py`), particle evolution traces (`plot_evolution_all.py`), particle inspection (`inspect_particles.py`). `plot_utils.py` provides shared plotting helpers used across these scripts.
- **`pipeline/scripts/`** — batch/dump inspection utilities (`inspect_batch.py`, `inspect_dump.py`).
