# Scalable HDF5 Chemistry Pipeline

This directory contains a standalone batch-parallel post-processing pipeline for PHANTOM SPH chemistry runs. It does not modify the existing trace and chemistry workflow under `code_trace/` and `code_chem/`.

## What it does

Each SLURM array task processes one particle batch end to end:

1. Discover the particle IDs directly from a PHANTOM dump with `sarracen`.
2. Exclude the first `n_boundary` IDs during discovery.
3. Select one batch by particle count, not by a pre-generated `particle_IDs.txt` file.
4. Run `phantomanalysis` serially in one call over the ordered configured dump list inside an isolated batch workspace.
5. Convert each generated `.phys` file into chemistry input.
6. Run the chemistry model for every configured chemistry type.
7. Convert `ev_output.dat` into one persistent `batch_XXXXX.h5` file.
8. Delete temporary `.phys`, `.txt`, `param`, `output`, and `ev_output` files after append.
9. Merge `batch_*.h5` into final `dump_XXXXX.h5` files after all batches finish.

Particle histories are stored as variable-length per-particle arrays in batch files (no NaN padding for shorter traces). During merge, each dump output includes only particles that have data for that timestep.

## Directory layout

- `src/analysis_trace.f90`: copied reference trace analysis source.
- `src/convert_ev_to_hdf5.f90`: Fortran HDF5 converter scaffold for future compiled use.
- `scripts/convert_phys_to_txt.py`: standalone `.phys -> chemistry input` converter.
- `scripts/run_batch_pipeline.py`: main batch runner.
- `scripts/append_ev_to_hdf5.py`: current operational HDF5 appender using `h5py` and `float64` datasets.
- `scripts/merge_batches.py`: merges batch files into per-dump HDF5 outputs.
- `slurm/run_batch.slurm`: one array task per batch.
- `slurm/merge_dumps.slurm`: final merge job.
- `workflow/run_batch.sh`: shell wrapper used by the batch job.
- `config/pipeline_config.yaml`: configurable paths and scaling parameters.

## Configuration

Edit `config/pipeline_config.yaml` before submitting:

- `paths.base_path`: shared filesystem root for all pipeline paths.
- `paths.phantom_dump_dir`: PHANTOM dump directory.
- `paths.av_dir`: directory containing `AV_XXXXX` files.
- `paths.phantomanalysis_binary`: compiled `phantomanalysis` binary to run.
- `paths.chemistry_model_root`: existing evolving-model runtime directory.
- `paths.batch_output_dir`: persistent `batch_XXXXX.h5` output location.
- `paths.final_output_dir`: final `dump_XXXXX.h5` output location.
- `paths.scratch_root`: per-batch scratch workspaces.

All other `paths.*` entries may be written relative to `paths.base_path`; they are resolved to absolute paths when loading the config.
- `processing.batch_size`: particles per job.
- `processing.n_batches`: number of jobs to expose.
- `processing.n_boundary`: number of low-ID particles to exclude during discovery. Keep this at `0` for small test runs and set it to values such as `5000` only when the selected dump actually contains that many removable boundary particles.
- `processing.chemistry_types`: `Crich`, `Orich`, or both.
- `processing.reuse_existing_trace_output`: when `true`, skip `phantomanalysis` for a batch if all expected `trace_output/<id>.phys` files already exist.

If both `batch_size` and `n_batches` are set, the pipeline processes the first `batch_size * n_batches` discovered particles. This makes small-scale testing cheap while keeping production layout predictable.

## HDF5 layout

Each batch job writes one file:

```text
batch_00042.h5
  /particles/id
  /trace/dumps/dump_00010/x
  /trace/dumps/dump_00010/y
  /trace/dumps/dump_00010/z
  /trace/dumps/dump_00010/r
  /trace/dumps/dump_00010/density
  /trace/dumps/dump_00010/temp
  /trace/dumps/dump_00010/av
  /trace/dumps/dump_00010/time
  /chemistry/Crich/dumps/dump_00010/co
  /chemistry/Orich/dumps/dump_00010/co
```

All numeric datasets are written as `float64`. Particle IDs are `int64`.

Merged dump files contain:

- `/particles/id`, `/particles/x`, `/particles/y`, `/particles/z`, `/particles/r`, `/particles/density`, `/particles/temp`, `/particles/av`, `/particles/time`
- Species directly under `/particles/<species>` when one chemistry type is configured
- Species under `/chemistry/<type>/particles/<species>` when multiple chemistry types are configured

## Running

### One-time (or when dump-range config changes): discover particles in parallel

Before submitting batch chemistry jobs, build the particle-ID cache with the dedicated SLURM workflow:

- `cd slurm`
- `./submit_discover_particle_ids.sh`
- optional throttle: `./submit_discover_particle_ids.sh --max-concurrent 256`

This runs one array task per dump, merges IDs, and writes discovery results back to `config/pipeline_config.yaml`:

- `paths.particle_ids_cache`
- `processing.discovered_particle_count_all`
- `processing.discovered_particle_count`
- `processing.discovered_batch_count`

`submit_batch.sh` then uses these cached values and no longer performs expensive live `sarracen` scans at submit time.

Important: cache now stores all discovered particle IDs (`particle_ids_all.npy`) and applies `processing.n_boundary` at runtime.
This means changing `n_boundary` in `pipeline_config.yaml` takes effect immediately on the next batch submission without recompiling.

Test configuration:

1. Set `batch_size: 1` and `n_batches: 10` in `config/pipeline_config.yaml`.
2. Submit with wrapper from the slurm directory:
  - `./submit_batch.sh`
  - optional throttle: `./submit_batch.sh --max-concurrent 256`
3. After all array tasks succeed, submit merge with:
  - `./submit_merge.sh`

Production configuration:

1. Set `batch_size` and `n_batches` to your production values.
2. Submit with `./submit_batch.sh` (the wrapper computes the array size automatically from discovered particles and batch layout).
3. Submit merge with `./submit_merge.sh` after batch completion.

### Logging

- Logs are written under `paths.scratch_root/logs/` with per-submission timestamped directories:
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/chem_hdf5_batch_%A_%a.out`
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/chem_hdf5_batch_%A_%a.err`
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/chem_hdf5_merge_%A.out`
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/chem_hdf5_merge_%A.err`

This follows the same submit-wrapper style used in `code_chem` and `code_trace` while keeping logs in the configured work area.

## Failure recovery

Batch recovery is file-based:

1. Delete the failed `batch_XXXXX.h5`.
2. Re-run only that batch index.
3. Leave all other batch files untouched.

## Notes on the Fortran converter

The requested `src/convert_ev_to_hdf5.f90` is included as the intended compiled HDF5 path, but the current environment does not provide `h5fc` or `h5pfc`. The working pipeline therefore uses `scripts/append_ev_to_hdf5.py` by default so the new architecture is runnable immediately. Once an HDF5 Fortran toolchain is available, the batch runner can be switched to a compiled converter without changing the batch layout or merge step.