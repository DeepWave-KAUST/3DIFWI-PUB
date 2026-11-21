#!/usr/bin/env python3
"""Step 2 of the Overthrust example: implicit FWI with frequency-selection source encoding.

The velocity is the linear starting model plus 500 m/s times a SIREN network fed by an
anisotropic multi-resolution hash encoding. At every iteration the 50 sources of one pool fire
in a single simulation, each on its own frequency component, and a generalized-correlation
(GCN) misfit compares the decoded steady-state responses with the observed coefficients source
by source, so no wavelet is estimated. Five cumulative bands (2-4 to 2-16 Hz, 200 iterations),
one GPU, or several GPUs with the model split into tiles. The observed coefficients come from
overthrust_forward.py, which runs once.

    python overthrust_invert.py --workdir overthrust
    python overthrust_invert.py --workdir overthrust --boundary-storage cpu    # smaller GPUs
    torchrun --standalone --nproc-per-node 4 overthrust_invert.py --workdir overthrust --dd 2x2

With --dd PYxPX the wavefield is split into PY x PX tiles, one GPU each (sweep's ModelParallel).
Every GPU renders the whole model from the same network; the misfit sums and the velocity
gradient are summed over the tiles, so every GPU takes the same optimizer step.

The state is saved after every band, and a rerun continues after the last finished band.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from overthrust_common import (BANDS, N_BINS, NATIVE_DH, NATIVE_SHAPE, SEED, TOTAL_EPOCHS, Band,
                               acquisition, data_files, make_solver, resample, stage_shape,
                               starting_model)

# Frequency-selection encoding.
N_POOLS = 8                     # 400 sources in 8 pools of 50; one pool fires per iteration
RAMP_S = 0.5                    # cosine ramp-up of the encoded sinusoids
GCN_EPS = 1e-12

# Network: anisotropic hash encoding + SIREN; v = clamp(v0 + 500 m/s * net, 1500, 6500).
HASH_LEVELS, HASH_FEATURES, HASH_LOG2_SIZE = 16, 2, 22
HASH_BASE_RES, HASH_FINEST_RES = (6, 7, 7), (186, 228, 228)     # per axis (z, y, x)
C2F_BASE_LEVELS, C2F_FINAL_LEVELS = 2, 12        # open 2 -> 12 levels over the whole cascade
SIREN_WIDTH, SIREN_HIDDEN_LAYERS, SIREN_OMEGA0 = 256, 2, 10.0
VP_STD, VP_MIN, VP_MAX = 500.0, 1500.0, 6500.0
LEARNING_RATE = 1e-4


# ========================================================== encoding & misfit ====
class ObservedTable:
    """Observed coefficients of one band: for every shot, its receivers and one value per bin."""

    def __init__(self, workdir: Path, band: Band, dev):
        parts = data_files(workdir, band)
        if not parts:
            raise FileNotFoundError(f"no observed data for {band.name}; run overthrust_forward.py first")
        self.rec_idx, self.D, self.dev = {}, {}, dev
        for p in parts:
            z = np.load(p)
            if not np.allclose(z["freqs"], band.freqs) or int(z["nt"]) != band.nt:
                raise ValueError(f"{p.name} was made for another band table")
            ptr, rec, D = z["ptr"], z["rec"], z["D"]
            for i, s in enumerate(z["shots"]):
                self.rec_idx[int(s)] = torch.as_tensor(rec[ptr[i]:ptr[i + 1]], dtype=torch.long, device=dev)
                self.D[int(s)] = np.ascontiguousarray(D[ptr[i]:ptr[i + 1]].T)   # (bins, receivers)

    def column(self, s: int, k: int):
        d = torch.as_tensor(self.D[s][k], device=self.dev)
        return d.real.float(), d.imag.float()


class PoolScheduler:
    """Spatially interleaved pools; a fresh permutation of the bins at every iteration."""

    def __init__(self, src_grid: np.ndarray, available, seed: int):
        _, first = np.unique(src_grid, axis=0, return_index=True)     # one source per grid cell
        keep = np.sort(first)
        keep = keep[np.isin(keep, np.asarray(sorted(available)))]
        order = keep[np.argsort(src_grid[keep, 1], kind="stable")]     # by y, then by index
        self.pools = [np.sort(order[p::N_POOLS]) for p in range(N_POOLS)]
        assert max(len(p) for p in self.pools) <= N_BINS
        self.rng = np.random.default_rng(seed)

    def draw(self, it: int):
        pool = self.pools[it % N_POOLS]
        return pool, self.rng.permutation(N_BINS)[:len(pool)]


def encoded_wavelet(band: Band, bins, dev) -> torch.Tensor:
    """(n_sources, nt) rows of ramp(t) * cos(2 pi f_b t)."""
    t = np.arange(band.nt, dtype=np.float64) * band.dt
    ramp = np.where(t < RAMP_S, 0.5 * (1.0 - np.cos(np.pi * t / RAMP_S)), 1.0)
    w = ramp[None] * np.cos(2 * np.pi * band.freqs[bins][:, None] * t[None])
    return torch.tensor(w, dtype=torch.float32, device=dev)


def gcn_misfit(record: torch.Tensor, pool, bins, band: Band, obs: ObservedTable, local=None):
    """Sum over the pool of 1 - |<u, d>| / (|u| |d|), u the decoded steady-state response.

    Domain-decomposed run: record holds only the receivers of this tile, local maps a global
    receiver index to its column (-1 elsewhere), and the four sums per source are added over the
    tiles before the misfit is formed. The backward then needs only this tile's part.
    """
    dev = record.device
    start = band.steady + band.slack
    y = record[0, :, :, 0].transpose(0, 1)[:, start:start + band.n_p]          # (receivers, n_p)
    m = np.arange(band.n_p, dtype=np.float64)
    ph = np.exp(-2j * np.pi * band.freqs[bins][:, None] * (start + m[None]) * band.dt) * (2.0 / band.n_p)
    er = torch.tensor(ph.real, dtype=torch.float32, device=dev)
    ei = torch.tensor(ph.imag, dtype=torch.float32, device=dev)
    part = torch.zeros(len(pool), 4, device=dev)
    for j, s in enumerate(pool):
        ri = obs.rec_idx[int(s)]
        dr, di = obs.column(int(s), int(bins[j]))
        if local is not None:
            col = local[ri]
            own = col >= 0
            ys, dr, di = y[col[own]], dr[own], di[own]
        else:
            ys = y[ri]
        ur, ui = ys @ er[j], ys @ ei[j]
        part[j, 0] = (ur * dr + ui * di).sum()
        part[j, 1] = (ui * dr - ur * di).sum()
        part[j, 2] = (ur ** 2 + ui ** 2).sum()
        part[j, 3] = (dr ** 2 + di ** 2).sum()
    if local is None:
        return _gcn(part)
    total = part.detach().clone()
    dist.all_reduce(total)                                       # sums over the tiles
    total.requires_grad_(True)
    loss = _gcn(total)
    (g,) = torch.autograd.grad(loss, total)
    return loss.detach(), part, g                                # backward: part.backward(g)


def _gcn(part: torch.Tensor) -> torch.Tensor:
    num = torch.sqrt(part[:, 0] ** 2 + part[:, 1] ** 2 + GCN_EPS)
    den = torch.sqrt(part[:, 2] + GCN_EPS) * torch.sqrt(part[:, 3] + GCN_EPS)
    return (1.0 - num / den).sum()


# ================================================================ network ====
class HashGrid(nn.Module):
    """Anisotropic multi-resolution hash encoding with coarse-to-fine level masking."""

    def __init__(self) -> None:
        super().__init__()
        self.L, self.F, self.T = HASH_LEVELS, HASH_FEATURES, 2 ** HASH_LOG2_SIZE
        b = [math.exp((math.log(HASH_FINEST_RES[d]) - math.log(HASH_BASE_RES[d])) / (self.L - 1))
             for d in range(3)]
        scales, strides, offsets, dense = [], [], [0], 0
        for level in range(self.L):
            s = [float(HASH_BASE_RES[d]) * b[d] ** level - 1.0 for d in range(3)]
            res = [int(math.ceil(v) + 1) for v in s]
            scales.append(s)
            prod = res[0] * res[1] * res[2]
            if prod <= self.T:                       # small enough to store densely
                n_entries = (prod + 7) // 8 * 8
                dense += 1
            else:
                n_entries = self.T
            offsets.append(offsets[-1] + n_entries)
            strides.append([1, res[0], res[0] * res[1]])
        self.dense_levels = dense
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.int64))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64))
        self.register_buffer("primes", torch.tensor([1, 2_654_435_761, 805_459_861], dtype=torch.long),
                             persistent=False)
        self.latents = nn.Parameter(torch.empty(offsets[-1], self.F).uniform_(-1.0e-4, 1.0e-4))
        corners = [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        self.register_buffer("corners", torch.tensor(corners, dtype=torch.float32))
        w = torch.zeros(self.L)
        w[:C2F_BASE_LEVELS] = 1.0
        self.register_buffer("level_weights", w, persistent=False)

    def set_progress(self, progress: float) -> float:
        """Open C2F_BASE_LEVELS -> C2F_FINAL_LEVELS levels linearly in progress; cosine frontier."""
        alpha = C2F_BASE_LEVELS + min(1.0, max(0.0, progress)) * (C2F_FINAL_LEVELS - C2F_BASE_LEVELS)
        w = [0.0 if alpha - i <= 0 else 1.0 if alpha - i >= 1 else 0.5 * (1 - math.cos(math.pi * (alpha - i)))
             for i in range(self.L)]
        self.level_weights.copy_(torch.tensor(w))
        return float(sum(w))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        n = pos.shape[0]
        ps = pos[None] * self.scales[:, None, :]                                # (L, n, 3)
        vert = (torch.floor(ps).unsqueeze(-2) + self.corners.view(1, 1, -1, 3)).to(torch.long)
        parts = []
        if self.dense_levels:
            parts.append((vert[:self.dense_levels] * self.strides[:self.dense_levels].view(-1, 1, 1, 3)).sum(-1))
        if self.dense_levels < self.L:
            mask = 0xFFFF_FFFF
            v = vert[self.dense_levels:]
            parts.append(((v[..., 0] & mask) ^ ((v[..., 1] * self.primes[1]) & mask)
                          ^ ((v[..., 2] * self.primes[2]) & mask)) & mask)
        idx = torch.cat(parts, 0) % self.T + self.offsets[:-1].view(self.L, 1, 1)
        idx = idx.clamp(0, self.latents.shape[0] - 1)
        frac = ps - torch.floor(ps)
        wts = ((1.0 - self.corners) + (2.0 * self.corners - 1.0) * frac.unsqueeze(-2)).clamp(0, 1).prod(-1)
        enc = (self.latents[idx] * wts.unsqueeze(-1)).sum(-2)                   # (L, n, F)
        return enc.permute(1, 0, 2).reshape(n, self.L * self.F) * self.level_weights.repeat_interleave(self.F)


class SineLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, *, first: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        bound = 1.0 / n_in if first else math.sqrt(6.0 / n_in) / SIREN_OMEGA0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(SIREN_OMEGA0 * self.linear(x))


class ImplicitVelocity(nn.Module):
    """v(x) = clamp(v0(x) + VP_STD * SIREN(hash(x)), VP_MIN, VP_MAX), x in the native frame."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = HashGrid()
        layers = [SineLayer(HASH_LEVELS * HASH_FEATURES, SIREN_WIDTH, first=True)]
        layers += [SineLayer(SIREN_WIDTH, SIREN_WIDTH, first=False) for _ in range(SIREN_HIDDEN_LAYERS)]
        out = nn.Linear(SIREN_WIDTH, 1, bias=False)
        with torch.no_grad():
            bound = math.sqrt(6.0 / SIREN_WIDTH) / SIREN_OMEGA0
            out.weight.uniform_(-bound, bound)
        self.mlp = nn.Sequential(*layers, out)
        self.base, self.axes = None, None

    def set_band(self, base: torch.Tensor, dh: float) -> None:
        """Render on a band's grid: node i of an axis sits at min(i dh, extent) / native length."""
        self.base = base
        self.axes = [torch.tensor(np.minimum(np.arange(n) * dh, (n0 - 1) * NATIVE_DH) / (n0 * NATIVE_DH),
                                  dtype=torch.float32, device=base.device)
                     for n, n0 in zip(base.shape, NATIVE_SHAPE)]

    def rows(self, z0: int, z1: int) -> torch.Tensor:
        c = torch.stack(torch.meshgrid(self.axes[0][z0:z1], self.axes[1], self.axes[2], indexing="ij"), -1)
        raw = self.mlp(self.encoder(c.reshape(-1, 3))).reshape(z1 - z0, *self.base.shape[1:])
        return (self.base[z0:z1] + raw * VP_STD).clamp(VP_MIN, VP_MAX)

    @torch.no_grad()
    def render(self) -> torch.Tensor:
        return torch.cat([self.rows(z, z + 1) for z in range(self.base.shape[0])])

    def backward_velocity(self, grad: torch.Tensor) -> None:
        """Push dJ/dv through the network one depth row at a time (renders each row again)."""
        for z in range(self.base.shape[0]):
            self.rows(z, z + 1).backward(grad[z:z + 1])


# ============================================================== inversion ====
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", default="overthrust", help="directory of the forward step (default ./overthrust)")
    ap.add_argument("--boundary-storage", default="gpu", choices=("gpu", "cpu"),
                    help="where the saved boundary is kept (cpu: for smaller GPUs, slower)")
    ap.add_argument("--dd", default=None, metavar="PYxPX",
                    help="split the model into PY x PX tiles, one GPU each; launch with torchrun "
                         "--nproc-per-node PY*PX (e.g. --dd 2x2 on four GPUs)")
    ap.add_argument("--bands", nargs="*", default=None, help=argparse.SUPPRESS)     # testing only
    ap.add_argument("--max-iters", type=int, default=0, help=argparse.SUPPRESS)     # testing only
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")
    from sweep.propagator.options import BoundarySaving
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.dd:
        py, px = (int(n) for n in args.dd.lower().split("x"))
        if py * px != world:
            raise SystemExit(f"--dd {args.dd} needs {py * px} processes (torchrun --nproc-per-node "
                             f"{py * px}), got {world}")
        from sweep.parallel import MeshTopology, ModelParallel, pad_to_mesh
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")
        mesh = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
    elif world > 1:
        raise SystemExit("several processes but no --dd PYxPX")
    log = print if rank == 0 else (lambda *a, **k: None)
    dev = torch.device("cuda", torch.cuda.current_device())
    workdir = Path(args.workdir)
    missing = [b.name for b in BANDS if (not args.bands or b.name in args.bands) and not data_files(workdir, b)]
    if missing:
        raise SystemExit(f"no observed data for {', '.join(missing)} in {workdir / 'data'}; "
                         "run overthrust_forward.py first")
    start_native = torch.from_numpy(starting_model(workdir)).to(dev)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = ImplicitVelocity().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-8, betas=(0.9, 0.999))
    curves = dict(loss=[], iter_s=[], peak_gb=[], band=[])
    done = -1
    for i, band in enumerate(BANDS):                                  # resume after the last band
        f = workdir / f"state_{band.name}.pt"
        if f.exists():
            st = torch.load(f, map_location=dev, weights_only=False)
            net.load_state_dict(st["net"])
            opt.load_state_dict(st["opt"])
            curves = st["curves"]
            done = i
    if done >= 0:
        log(f"[invert] resuming after {BANDS[done].name}", flush=True)
    offset = sum(b.epochs for b in BANDS[:done + 1])
    for bi, band in enumerate(BANDS):
        if bi <= done or (args.bands and band.name not in args.bands):
            continue
        obs = ObservedTable(workdir, band, dev)
        src, rec, _ = acquisition(band.dh)
        sched = PoolScheduler(src, obs.rec_idx.keys(), seed=SEED + 17)
        base = resample(start_native, band.dh)
        net.set_band(base, band.dh)
        for g in opt.param_groups:
            g["lr"] = LEARNING_RATE
        tail = None if band.tail_s is None else band.n_p + int(math.ceil(band.tail_s / band.dt))
        memory = BoundarySaving(storage=args.boundary_storage, storage_dtype="int8", tail_steps=tail)
        if args.dd:
            solver = ModelParallel(make_solver(band, memory, shape=tuple(pad_to_mesh(base, mesh).shape)), mesh)
        else:
            solver = make_solver(band, memory)
        epochs = min(band.epochs, args.max_iters) if args.max_iters else band.epochs
        log(f"[invert] {band.name}: grid {stage_shape(band.dh)} dh {band.dh:g} m, dt {band.dt:g} s, "
              f"nt {band.nt}, {band.freqs[0]:.4f}-{band.freqs[-1]:.4f} Hz, {len(obs.D)} shots, {epochs} iterations, "
              f"boundary {args.boundary_storage}/int8, tail_steps {tail}"
            f"{f', {py}x{px} tiles' if args.dd else ''}", flush=True)
        for it in range(epochs):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            levels = net.encoder.set_progress((offset + it) / max(1, TOTAL_EPOCHS - 1))
            pool, bins = sched.draw(it)
            opt.zero_grad()
            vp = net.render().requires_grad_(True)
            wav = encoded_wavelet(band, bins, dev)
            if args.dd:
                record = solver(wav, src[pool][None], rec[None], models=[pad_to_mesh(vp, mesh)])
                local = torch.full((len(rec),), -1, dtype=torch.long, device=dev)
                own = torch.as_tensor(solver.own_receiver_indices, dtype=torch.long, device=dev)
                local[own] = torch.arange(len(own), device=dev)
                loss, part, g = gcn_misfit(record, pool, bins, band, obs, local=local)
                part.backward(g)
                del record
                dist.all_reduce(vp.grad)                  # every tile's part of dJ/dv
            else:
                record = solver(wav, src[pool][None], rec[None], models=[vp])
                loss = gcn_misfit(record, pool, bins, band, obs)
                loss.backward()
                del record
            net.backward_velocity(vp.grad)
            if args.dd:                                   # keep the replicas identical: the network
                for prm in net.parameters():              # backward uses atomics, so its last bits
                    if prm.grad is not None:              # can differ from GPU to GPU
                        dist.all_reduce(prm.grad)
                        prm.grad /= world
            opt.step()
            curves["loss"].append(float(loss.detach()) / len(pool))
            curves["iter_s"].append(time.time() - t0)
            curves["peak_gb"].append(torch.cuda.max_memory_allocated() / 2 ** 30)
            curves["band"].append(bi)
            if not math.isfinite(curves["loss"][-1]):
                raise FloatingPointError(f"{band.name} iteration {it}: misfit is not finite")
            log(f"[invert] {band.name} it {it:3d} pool {it % N_POOLS}  mean(1-GCN)={curves['loss'][-1]:.5f}  "
                  f"levels {levels:5.2f}  iter_s={curves['iter_s'][-1]:.1f}  peak_gb={curves['peak_gb'][-1]:.2f}",
                  flush=True)
        offset += band.epochs
        del solver, obs, vp
        torch.cuda.empty_cache()
        net.set_band(start_native, NATIVE_DH)          # every band's model is saved on the native grid
        vp_native = net.render().cpu().numpy()
        if rank == 0:
            np.save(workdir / f"inverted_vp_{band.name}.npy", vp_native)
            if epochs == band.epochs:
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), curves=curves),
                           workdir / f"state_{band.name}.pt")
    if rank == 0:
        np.savez(workdir / "curves.npz", **{k: np.asarray(v) for k, v in curves.items()})
    log("[invert] done", flush=True)
    if args.dd:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
