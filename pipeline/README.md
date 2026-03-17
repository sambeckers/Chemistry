# Scalable HDF5 Chemistry Pipeline

This directory contains a standalone batch-parallel post-processing pipeline for PHANTOM SPH chemistry runs. It does not modify the existing trace and chemistry workflow under `code_trace/` and `code_chem/`.

## What it does

The workflow runs as: discovery -> tracing -> batch chemistry -> merge.

1. Discover the particle IDs directly from PHANTOM dumps with `sarracen`.
2. Exclude the first `n_boundary` IDs during discovery.
3. Run `phantomanalysis` once over the ordered configured dump list.
4. Concatenate generated per-particle `.phys` outputs into per-batch `tracing_batch_XXXXX.h5` files.
5. For each chemistry batch, load tracing rows from `tracing_batch_XXXXX.h5` and convert to evolving-model input.
6. Run the chemistry model for every configured chemistry type.
7. Convert `ev_output.dat` into one persistent `batch_XXXXX.h5` file.
8. Build/reuse a dump-number -> real-time(seconds) map from PHANTOM metadata (`time * utime`).
9. Map each chemistry row to exact dump numbers using EV `TIME` values (years) with fixed-precision second keys; for legacy low-precision EV outputs, fall back to deterministic row-order suffix alignment.
10. Delete temporary `.txt`, `param`, `output`, and `ev_output` files after append.
11. Merge `batch_*.h5` into final `dump_XXXXX.h5` files after all batches finish.

Particle histories are stored as variable-length per-particle arrays in batch files (no NaN padding for shorter traces). During merge, each dump output includes only particles that have data for that timestep.

## Directory layout

- `src/analysis_trace.f90`: legacy reference copy (runtime tracing uses `paths.analysis_trace_source` in the Phantom tree).
- `src/convert_ev_to_hdf5.f90`: Fortran HDF5 converter scaffold for future compiled use.
- `scripts/convert_phys_to_txt.py`: standalone `.phys -> chemistry input` converter.
- `scripts/run_tracing.py`: one-shot tracing runner (`phantomanalysis` once, then pack per-batch tracing HDF5).
- `scripts/run_batch_pipeline.py`: main batch runner.
- `scripts/append_ev_to_hdf5.py`: current operational HDF5 appender using `h5py` and `float64` datasets.
- `scripts/merge_batches.py`: merges batch files into per-dump HDF5 outputs.
- `slurm/run_tracing.slurm`: tracing job (runs between discovery and batch jobs).
- `slurm/run_batch.slurm`: one array task per batch.
- `slurm/merge_dumps.slurm`: final merge job.
- `workflow/batch.sh`: unified script for submit mode and per-task run mode.
- `workflow/submit_discover_particle_ids.sh`: particle-ID discovery submit wrapper.
- `workflow/submit_merge.sh`: merge submit wrapper.
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
- `paths.dump_time_map_cache`: JSON cache of dump-number -> real-time(seconds) mapping.
- `paths.trace_batches_dir`: legacy fallback directory for `tracing_batch_XXXXX.h5` files (binary traces are primary).
- `paths.trace_metadata_file`: metadata YAML written by tracing step.
- `paths.analysis_trace_source`: source of `analysis_trace.f90` in your Phantom tree (used for tracing provenance).

All other `paths.*` entries may be written relative to `paths.base_path`; they are resolved to absolute paths when loading the config.
- `processing.time_key_decimals`: fixed decimal precision used for second-based time keys when matching EV `TIME` rows to dumps.
- `processing.discovery_mode`: particle-discovery submission mode; use `serial` for one-job scan or `array` for dump-parallel workers.
- `processing.batch_size`: particles per job.
- `processing.n_batches`: number of jobs to expose.
- `processing.n_boundary`: number of low-ID particles to exclude during discovery. Keep this at `0` for small test runs and set it to values such as `5000` only when the selected dump actually contains that many removable boundary particles.
- `processing.chemistry_types`: `Crich`, `Orich`, or both.

If both `batch_size` and `n_batches` are set, the pipeline processes the first `batch_size * n_batches` discovered particles. This makes small-scale testing cheap while keeping production layout predictable.

## HDF5 layout

Each batch job writes one file with per-particle variable-length vectors and explicit dump alignment:

```text
batch_00042.h5
  /particles/id
  /trace/particles/row_count
  /trace/particles/dump_number
  /trace/particles/x
  /trace/particles/y
  /trace/particles/z
  /trace/particles/r
  /trace/particles/density
  /trace/particles/temp
  /trace/particles/av
  /trace/particles/time
  /chemistry/Crich/particles/row_count
  /chemistry/Crich/particles/dump_number
  /chemistry/Crich/particles/co
  /chemistry/Orich/particles/row_count
  /chemistry/Orich/particles/dump_number
  /chemistry/Orich/particles/co
```

All numeric value datasets are written as `float64`. Particle IDs are `int64`. Per-row dump indices are `int32`.

Merged alignment now prefers per-particle `dump_number` vectors, avoiding row-index assumptions for short traces and late-start particles.

Merged dump files contain:

- `/particles/id`, `/particles/x`, `/particles/y`, `/particles/z`, `/particles/r`, `/particles/density`, `/particles/temp`, `/particles/av`, `/particles/time`
- Species directly under `/particles/<species>` when one chemistry type is configured
- Species under `/chemistry/<type>/particles/<species>` when multiple chemistry types are configured

## Running

### One-time (or when dump-range config changes): discover particles in parallel

Before submitting batch chemistry jobs, build the particle-ID cache with the dedicated SLURM workflow:

- `cd <pipeline-root>`
- `./workflow/submit_discover_particle_ids.sh`
- optional throttle: `./workflow/submit_discover_particle_ids.sh --max-concurrent 256`
- optional serial mode: `./workflow/submit_discover_particle_ids.sh --serial`

This runs one array task per dump, merges IDs, and writes discovery results back to `config/pipeline_config.yaml`:

- `paths.particle_ids_cache`
- `paths.dump_time_map_cache`
- `processing.discovered_particle_count_all`
- `processing.discovered_particle_count`
- `processing.discovered_batch_count`

`workflow/batch.sh` then submits tracing automatically before chemistry batches and no longer performs expensive live `sarracen` scans at submit time.

Important: cache now stores all discovered particle IDs (`particle_ids_all.npy`) and applies `processing.n_boundary` at runtime.
This means changing `n_boundary` in `pipeline_config.yaml` takes effect immediately on the next batch submission without recompiling.

Test configuration:

1. Set `batch_size: 1` and `n_batches: 10` in `config/pipeline_config.yaml`.
2. Submit with unified workflow wrapper:
  - `./workflow/batch.sh`
  - optional throttle: `./workflow/batch.sh --max-concurrent 256`
3. After all array tasks succeed, submit merge with:
  - `./workflow/submit_merge.sh`

Production configuration:

1. Set `batch_size` and `n_batches` to your production values.
2. Submit with `./workflow/batch.sh` (the wrapper computes the array size automatically from discovered particles and batch layout).
3. Submit merge with `./workflow/submit_merge.sh` after batch completion.

### Logging

- Logs are written under `paths.scratch_root/logs/` with per-submission timestamped directories:
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/batch_%A_%a.out`
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/batch_%A_%a.err`
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/merge_%A.out`
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/merge_%A.err`
  - `logs_%A_YYYY-mm-dd_HH-MM-SS/batch_run_summary.txt` (auto-generated after array completion)

When you submit with `workflow/batch.sh`, the wrapper now submits a dependent summary job that runs after the batch array finishes (`afterany`).
The summary file contains:

- Per-batch runtime by key pipeline process (trace stage, chemistry model, append, total, and other stages)
- Warning/error line counts and sampled issue lines from each batch `*.out` and `*.err`
- Total run wall time across all batches (first task start to last task end)

This follows the same submit-wrapper style used in `code_chem` and `code_trace` while keeping logs in the configured work area.

## Failure recovery

Batch recovery is file-based:

1. Delete the failed `batch_XXXXX.h5`.
2. Re-run only that batch index.
3. Leave all other batch files untouched.

## Notes on the Fortran converter

The requested `src/convert_ev_to_hdf5.f90` is included as the intended compiled HDF5 path, but the current environment does not provide `h5fc` or `h5pfc`. The working pipeline therefore uses `scripts/append_ev_to_hdf5.py` by default so the new architecture is runnable immediately. Once an HDF5 Fortran toolchain is available, the batch runner can be switched to a compiled converter without changing the batch layout or merge step.

## EV time precision

Strict EV-time matching relies on chemistry model outputs providing sufficiently precise `TIME` values in years and using the same year-to-seconds convention as the dump-time map (`31557600` seconds per year).

The source in `evolving_model/inputmodel.f` now writes `TIME` with higher precision (`1PE20.12`) and uses `YR2SEC = 3.15576e+07` for this export path. Rebuild the chemistry executable after updating these sources, otherwise legacy outputs with coarse `TIME` precision will use the row-order fallback alignment.