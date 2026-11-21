# User guide

The example runs in three steps. Each step is one script; all three take a work directory
(`--workdir`, default `./overthrust`) that holds the inputs and outputs of the whole run.

```
overthrust_forward.py  ->  WORKDIR/data/obs_B*.npz, starting_model.json
overthrust_invert.py   ->  WORKDIR/inverted_vp_B*.npy, state_B*.pt, curves.npz
overthrust_figures.py  ->  WORKDIR/summary.txt, overthrust_*.png
```

## Requirements

- Linux x86_64 with an NVIDIA GPU (CUDA 12 driver or newer) for steps 1 and 2; step 3 runs on a CPU.
- GPU memory: 48 GB for the finest band on one GPU with the default settings, or 17 GB with
  `--boundary-storage cpu` (then about 55 GB of host memory); at most 28 GB per GPU with `--dd 2x2`.
- Disk: about 21 GB in the work directory (the 531 MB model file and 20 GB of observed data).
- Python 3.9 or newer with the packages in `requirements.txt`.

## Step 1: `overthrust_forward.py`

Models the observed data through the true model, once.

| option | default | meaning |
|---|---|---|
| `--workdir DIR` | `overthrust` | output directory |

Environment variable `OVERTHRUST_MODEL=/path/to/overthrust_3D_true_model.h5` uses an existing
copy of the model instead of downloading it.

Inputs: the SEG/EAGE Overthrust model, downloaded from Zenodo on first use and checked against
its md5.

Outputs:

- `WORKDIR/overthrust_3D_true_model.h5`: the downloaded model (squared slowness, with 500 m of water on top).
- `WORKDIR/starting_model.json`: the coefficients `a`, `b` of the linear starting model
  v(z) = a + b z, fitted to the lateral mean of the true model.
- `WORKDIR/data/obs_<band>_r<rank>of<world>.npz`: for each band, the observed coefficients. Each
  file holds the shots handled by one GPU: `shots` (shot indices), `rec` and `ptr` (the receivers
  kept for each shot, in compressed-row form), `D` (complex coefficients, one per receiver and
  frequency component), `freqs` and `nt` (the band it was made for).

Several GPUs: `torchrun --standalone --nproc-per-node N overthrust_forward.py --workdir DIR`.
Each GPU takes every N-th shot. A band whose files are complete is skipped, so a rerun continues
an interrupted run.

## Step 2: `overthrust_invert.py`

Runs the five-band inversion on one GPU, or on several with `--dd`.

| option | default | meaning |
|---|---|---|
| `--workdir DIR` | `overthrust` | directory written by step 1 |
| `--boundary-storage {gpu,cpu}` | `gpu` | where the wavefield boundary saved for the backward pass is kept; `cpu` needs less GPU memory and is slower |
| `--dd PYxPX` | off | split the model into PY x PX tiles, one GPU each (domain decomposition); launch with `torchrun --standalone --nproc-per-node PY*PX` |

With `--dd`, every GPU renders the whole model from the same network, sweep's `ModelParallel`
runs the wave equation on its tile, the per-source misfit sums and the velocity gradient are
added over the tiles, and every GPU takes the same optimizer step. Rank 0 writes the outputs.
Example: `torchrun --standalone --nproc-per-node 4
overthrust_invert.py --workdir overthrust --dd 2x2`.

Inputs: `WORKDIR/data/obs_*.npz` and `WORKDIR/starting_model.json` from step 1.

Outputs:

- `WORKDIR/inverted_vp_<band>.npy`: the velocity (m/s) after each band, float32, shape
  (187, 801, 801) = (nz, ny, nx) on the 25 m grid of the original model.
- `WORKDIR/state_<band>.pt`: network and optimizer state after each band. A rerun loads the last
  one and continues with the next band.
- `WORKDIR/curves.npz`: per iteration, `loss` (mean of 1 - GCN over the sources of the pool),
  `iter_s` (seconds), `peak_gb` (peak GPU memory) and `band` (band index).

## Step 3: `overthrust_figures.py`

Compares the inverted model with the true model and draws the figures. No GPU needed.

| option | default | meaning |
|---|---|---|
| `--workdir DIR` | `overthrust` | directory written by step 2 |
| `--band B1..B5` | `B5` | which band's model to evaluate (B5 is the final model) |

Outputs:

- `WORKDIR/summary.txt`: r.m.s. error of the starting and inverted models against the true model,
  for the whole volume and by depth interval, and the misfit at the first and last iteration.
- `WORKDIR/overthrust_models.png`: inline and crossline sections and a depth slice of the true,
  starting and inverted models.
- `WORKDIR/overthrust_profiles.png`: vertical profiles at three locations.
- `WORKDIR/overthrust_misfit.png`: the misfit over the 200 iterations.

## Changing the settings

All settings are constants at the top of the scripts:

- `overthrust_common.py`: model download, acquisition (`SRC_1D`, `REC_1D`, `MIN_OFFSET`,
  `MAX_OFFSET`), solver order and absorbing layer (`SPATIAL_ORDER`, `PML_CELLS`), the number of
  frequency components per band (`N_BINS`), and the band table `BANDS` (grid spacing, time step,
  iterations, analysis window, lowest frequency bin, ring-up and slack samples, backward
  truncation, shots per forward simulation).
- `overthrust_forward.py`: the source wavelet (`RICKER_FM`, `RICKER_T0`) and its low-pass filter.
- `overthrust_invert.py`: source pools (`N_POOLS`), the network (`HASH_*`, `SIREN_*`), the
  coarse-to-fine schedule of the hash levels (`C2F_BASE_LEVELS`, `C2F_FINAL_LEVELS`), the velocity
  scaling and bounds (`VP_STD`, `VP_MIN`, `VP_MAX`) and the learning rate.

After changing `BANDS` or the acquisition, delete `WORKDIR/data` so step 1 models the data again;
the inversion checks that the observed files match the band table and stops if they do not.
