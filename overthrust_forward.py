#!/usr/bin/env python3
"""Step 1 of the Overthrust example: model the observed data once.

Every one of the 400 shots is forward-modelled through the true Overthrust model on each band's
grid, with a Ricker wavelet low-passed at 1.5x the band's top frequency. Only the DTFT of each
record at the band's 50 frequency components is kept, for the receivers 0.5-12 km from the shot:
one table per band, about 20 GB in total under WORKDIR/data. The inversion
(overthrust_invert.py) reads these tables and never models the observed data again.

    python overthrust_forward.py --workdir overthrust
    torchrun --standalone --nproc-per-node 4 overthrust_forward.py --workdir overthrust   # 4 GPUs

The shots are independent, so each GPU takes every n-th shot. A band whose table exists is
skipped, so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch

from overthrust_common import (BANDS, N_BINS, acquisition, data_files, load_true_model, make_solver,
                               model_file, resample, stage_shape, starting_model)

RICKER_FM, RICKER_T0 = 8.0, 0.15          # Hz, s
LOWPASS_ORDER, LOWPASS_FACTOR = 6, 1.5     # causal Butterworth at 1.5x the band's top frequency


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", default="overthrust", help="output directory (default ./overthrust)")
    ap.add_argument("--bands", nargs="*", default=None, help=argparse.SUPPRESS)     # testing only
    ap.add_argument("--max-shots", type=int, default=0, help=argparse.SUPPRESS)     # testing only
    args = ap.parse_args()
    from scipy.signal import butter, sosfilt
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda", torch.cuda.current_device())
    workdir = Path(args.workdir)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo", timeout=datetime.timedelta(hours=3))
    if rank == 0:
        model_file(workdir)           # downloads the true model on first use
        starting_model(workdir)       # writes starting_model.json for the inversion
    if world > 1:
        dist.barrier()                # the other ranks read the model once it is there
    true = None
    for band in BANDS:
        if args.bands and band.name not in args.bands:
            continue
        if data_files(workdir, band):
            print(f"[forward r{rank}] {band.name}: observed data exist, skipping", flush=True)
            continue
        out_f = workdir / "data" / f"obs_{band.name}_r{rank:02d}of{world:02d}.npz"
        if out_f.exists():
            print(f"[forward r{rank}] {band.name}: {out_f.name} exists, skipping", flush=True)
            continue
        if true is None:
            true = torch.from_numpy(load_true_model(workdir))
        vp_true = resample(true.to(dev), band.dh)
        solver = make_solver(band)
        src, rec, keep = acquisition(band.dh)
        mine = np.arange(rank, len(src), world)
        if args.max_shots:
            mine = mine[:args.max_shots]
        t = np.arange(band.nt) * band.dt
        a = np.pi * RICKER_FM * (t - RICKER_T0)
        f_cut = round(LOWPASS_FACTOR * band.freqs.max(), 3)
        sos = butter(LOWPASS_ORDER, f_cut, btype="low", fs=1.0 / band.dt, output="sos")
        wav = torch.tensor(sosfilt(sos, (1 - 2 * a * a) * np.exp(-a * a)), dtype=torch.float32, device=dev)
        ph = 2 * np.pi * torch.tensor(band.freqs, dtype=torch.float64, device=dev)[:, None] \
            * torch.tensor(t, dtype=torch.float64, device=dev)[None]
        kr, ki = torch.cos(ph), -torch.sin(ph)               # (bins, nt) fp64 DTFT kernel
        print(f"[forward r{rank}] {band.name}: grid {stage_shape(band.dh)} dh {band.dh:g} m, dt {band.dt:g} s, "
              f"nt {band.nt}, bins {band.freqs[0]:.4f}-{band.freqs[-1]:.4f} Hz, low-pass {f_cut} Hz, "
              f"{len(mine)} shots", flush=True)
        D, recs, ptr = [], [], [0]
        t0 = time.time()
        for i in range(0, len(mine), band.shot_batch):
            sb = mine[i:i + band.shot_batch]
            with torch.no_grad():
                rec_b = solver(wav, src[sb], np.repeat(rec[None], len(sb), 0), models=[vp_true])[..., 0]
                if not bool(torch.isfinite(rec_b).all()):
                    raise FloatingPointError(f"non-finite record, shots {sb.tolist()}")
                for j, s in enumerate(sb):
                    dr = torch.empty((N_BINS, len(rec)), dtype=torch.float64, device=dev)
                    di = torch.empty_like(dr)
                    for c0 in range(0, len(rec), 8192):
                        x = rec_b[j, :, c0:c0 + 8192].double()
                        dr[:, c0:c0 + 8192] = kr @ x
                        di[:, c0:c0 + 8192] = ki @ x
                    k = np.nonzero(keep[s])[0]
                    D.append(torch.complex(dr, di).T[k].to(torch.complex64).cpu().numpy())
                    recs.append(k.astype(np.int32))
                    ptr.append(ptr[-1] + len(k))
            del rec_b
            print(f"[forward r{rank}] {band.name}: {min(i + band.shot_batch, len(mine))}/{len(mine)} shots, "
                  f"{time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB",
                  flush=True)
        tmp = out_f.with_suffix(".tmp.npz")
        np.savez(tmp, shots=mine, ptr=np.asarray(ptr, np.int64), rec=np.concatenate(recs),
                 D=np.concatenate(D), freqs=band.freqs, nt=band.nt, dt=band.dt)
        tmp.rename(out_f)
        del solver, vp_true
        torch.cuda.empty_cache()
    print(f"[forward r{rank}] done", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
