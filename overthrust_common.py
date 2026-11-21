"""Settings shared by the Overthrust forward modelling and inversion scripts.

The 3-D SEG/EAGE Overthrust example of "Three Ways to Afford 3D Implicit FWI" (Wang and
Alkhalifah, synthetic example): the model, the acquisition, the five-band cascade and the solver.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

NATIVE_SHAPE = (187, 801, 801)          # (nz, ny, nx) of the SEG/EAGE Overthrust model
NATIVE_DH = 25.0                        # m
SEED = 20260929

# Acquisition, in metres. Sources and receivers lie on the surface, in x-major order.
SRC_1D = 500.0 + 1000.0 * np.arange(20)          # 20 x 20 = 400 sources, 1 km apart
REC_1D = 100.0 * np.arange(1, 200)               # 199 x 199 = 39,601 receivers, 100 m apart
MIN_OFFSET, MAX_OFFSET = 500.0, 12000.0           # each source keeps the receivers 0.5-12 km away

# Physics: 4th-order acoustic wave equation, free surface on top, 20 CPML cells elsewhere.
SPATIAL_ORDER, PML_CELLS = 4, 20

N_BINS = 50                      # frequency components per band

# The SEG/EAGE Overthrust model (CC-BY-4.0; Aminzadeh, Brac and Kunz, 1997). The SEG download is no longer
# public; this copy on Zenodo (doi:10.5281/zenodo.4252588) stores squared slowness (s^2/km^2) on (z, x, y)
# with 500 m of water added on top.
MODEL_URL = "https://zenodo.org/records/4252588/files/overthrust_3D_true_model.h5"
MODEL_MD5 = "ae4082cdcebbc7714eedd382d7925804"
MODEL_WATER_ROWS = 20


@dataclass(frozen=True)
class Band:
    name: str
    dh: float            # grid spacing (m)
    dt: float            # time step (s)
    epochs: int          # iterations
    n_p: int             # probe (analysis) window in samples; the bins are exact DFT bins of it
    k_lo: int            # lowest bin; the band uses bins k_lo .. k_lo + N_BINS - 1
    steady: int          # ring-up samples before the window
    slack: int           # spare samples between ring-up and window
    tail_s: float | None  # truncated backward: keep the window + tail_s of the record, else all
    shot_batch: int      # shots per simulation in the forward modelling

    @property
    def nt(self) -> int:
        return self.steady + self.slack + self.n_p

    @property
    def freqs(self) -> np.ndarray:
        return np.arange(self.k_lo, self.k_lo + N_BINS) / (self.n_p * self.dt)


BANDS = (
    Band("B1", 100.0, 0.007, 60, 3500, 49, 1429, 286, None, 8),         # 2.00-4.00 Hz
    Band("B2", 50.0, 0.0035, 40, 3500, 24, 2857, 571, None, 8),         # 1.96-5.96 Hz
    Band("B3", 50.0, 0.0035, 40, 2333, 16, 2857, 571, None, 8),         # 1.96-7.96 Hz
    Band("B4", 100.0 / 3.0, 0.00225, 30, 2178, 10, 4444, 889, 6.0, 5),  # 2.04-12.04 Hz
    Band("B5", 25.0, 0.00175, 30, 2000, 7, 5714, 1143, 6.0, 4),         # 2.00-16.00 Hz
)
TOTAL_EPOCHS = sum(b.epochs for b in BANDS)


def stage_shape(dh: float) -> tuple[int, int, int]:
    """Points at i*dh that cover the native extent (the last may lie past it by under dh)."""
    return tuple(max(0, math.ceil((n - 1) * NATIVE_DH / dh - 1e-3)) + 1 for n in NATIVE_SHAPE)


def resample(native: torch.Tensor, dh: float) -> torch.Tensor:
    """A native-grid field sampled at i*dh: separable linear interpolation, edge value past it."""
    out = native
    for ax, n_dst in enumerate(stage_shape(dh)):
        n_src = native.shape[ax]
        if n_dst == n_src and abs(dh - NATIVE_DH) < 1e-9 * NATIVE_DH:
            continue
        c = np.arange(n_dst, dtype=np.float64) * (dh / NATIVE_DH)
        r = np.rint(c)
        c[np.abs(c - r) <= 1e-6] = r[np.abs(c - r) <= 1e-6]
        c = np.clip(c, 0.0, n_src - 1.0)
        i0 = np.minimum(np.floor(c).astype(np.int64), n_src - 1)
        i1 = np.minimum(i0 + 1, n_src - 1)
        view = [1] * out.ndim
        view[ax] = n_dst
        w = torch.as_tensor(c - i0, dtype=out.dtype, device=out.device).reshape(view)
        lo = out.index_select(ax, torch.as_tensor(i0, device=out.device))
        hi = out.index_select(ax, torch.as_tensor(i1, device=out.device))
        out = lo + (hi - lo) * w
    return out.contiguous()


def acquisition(dh: float):
    """Source and receiver grid indices (x, y, z) on a band's grid, and the receivers each keeps."""
    sx, sy = np.meshgrid(SRC_1D, SRC_1D, indexing="ij")
    rx, ry = np.meshgrid(REC_1D, REC_1D, indexing="ij")
    src = np.stack([sx.ravel(), sy.ravel()], -1)
    rec = np.stack([rx.ravel(), ry.ravel()], -1)
    off = np.hypot(rec[None, :, 0] - src[:, None, 0], rec[None, :, 1] - src[:, None, 1])
    keep = (off >= MIN_OFFSET) & (off <= MAX_OFFSET)

    def idx(m):
        return np.concatenate([np.rint(m / dh).astype(np.int32), np.zeros((len(m), 1), np.int32)], 1)
    return idx(src), idx(rec), keep


def model_file(workdir: Path) -> Path:
    """$OVERTHRUST_MODEL if set, else WORKDIR/overthrust_3D_true_model.h5, downloaded once (531 MB)."""
    if os.environ.get("OVERTHRUST_MODEL"):
        return Path(os.environ["OVERTHRUST_MODEL"])
    f = Path(workdir) / "overthrust_3D_true_model.h5"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        print(f"downloading the Overthrust model (531 MB) from {MODEL_URL}", flush=True)
        urllib.request.urlretrieve(MODEL_URL, tmp)
        md5 = hashlib.md5()
        with open(tmp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                md5.update(chunk)
        if md5.hexdigest() != MODEL_MD5:
            tmp.unlink()
            raise RuntimeError(f"{MODEL_URL}: md5 {md5.hexdigest()}, expected {MODEL_MD5}")
        tmp.rename(f)
    return f


def load_true_model(workdir: Path) -> np.ndarray:
    """The SEG/EAGE Overthrust P-wave velocity in m/s, (nz, ny, nx) = (187, 801, 801), without the water."""
    import h5py
    with h5py.File(model_file(workdir), "r") as h:
        m = h["m"][MODEL_WATER_ROWS:].astype(np.float64)
    vp = np.ascontiguousarray((1000.0 / np.sqrt(m)).astype(np.float32).transpose(0, 2, 1))
    assert vp.shape == NATIVE_SHAPE, vp.shape
    return vp


def starting_model(workdir: Path) -> np.ndarray:
    """v(z) = a + b z, the least-squares fit to the lateral mean of the true model."""
    f = Path(workdir) / "starting_model.json"
    if f.exists():
        p = json.loads(f.read_text())
    else:
        prof = load_true_model(workdir).mean(axis=(1, 2))
        b, a = np.polyfit(np.arange(NATIVE_SHAPE[0]) * NATIVE_DH, prof, 1)
        p = dict(a=float(a), b=float(b))
        f.write_text(json.dumps(p, indent=1))
    z = np.arange(NATIVE_SHAPE[0]) * NATIVE_DH
    v = (p["a"] + p["b"] * z).astype(np.float32)
    return np.broadcast_to(v[:, None, None], NATIVE_SHAPE).copy()


def make_solver(band: Band, memory=None, shape=None):
    """sweep-solver's 3-D acoustic propagator on a band's grid (compiled CUDA backend).

    shape overrides the grid, for the domain-decomposed run whose model is padded to the tile grid.
    """
    from sweep.equations.acoustic3d import Acoustic3D
    from sweep.propagator.torch import PropTorch
    dev = torch.device("cuda", torch.cuda.current_device())
    eq = Acoustic3D(spatial_order=SPATIAL_ORDER, device=dev, backend="torch")
    kw = dict(memory=memory) if memory is not None else {}
    return PropTorch(eq, shape=shape or stage_shape(band.dh), dev=dev, dh=band.dh, dt=band.dt, nt=band.nt,
                     abcn=PML_CELLS, free_surface=True, pml_type="cpmlr",
                     source_type=["h1"], receiver_type=["h1"], backend="torch", impl="c", **kw)


def data_files(workdir: Path, band: Band) -> list[Path]:
    """The observed-data parts of a band written by one complete forward run, else []."""
    sets: dict[int, dict[int, Path]] = {}
    for p in (Path(workdir) / "data").glob(f"obs_{band.name}_r*of*.npz"):
        m = re.fullmatch(rf"obs_{band.name}_r(\d+)of(\d+)\.npz", p.name)
        if m:
            sets.setdefault(int(m[2]), {})[int(m[1])] = p
    for world in sorted(sets):
        if len(sets[world]) == world:
            return [sets[world][r] for r in range(world)]
    return []
