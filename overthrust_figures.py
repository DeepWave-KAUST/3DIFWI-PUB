#!/usr/bin/env python3
"""Step 3 of the Overthrust example: error against the true model and the figures (no GPU).

    python overthrust_figures.py --workdir overthrust

Writes WORKDIR/summary.txt (r.m.s. error of the starting and inverted models against the true
model, whole volume and by depth) and three figures: sections and a depth slice of the true,
starting and inverted models, vertical profiles at three locations, and the misfit curve.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from overthrust_common import BANDS, NATIVE_DH, NATIVE_SHAPE, TOTAL_EPOCHS, load_true_model, starting_model

# The paper's velocity colour map (sweep_vp), 256 RGB entries.
_SWEEP_VP_HEX = """
fbeefdfbe5fdfbdefdfad7fcf9cefcf8c6fbf8befaf7b6faf7aff9f6a7f9f59ff9f496f8f48ef8f386f7f37df7f276f7
f16df7f067f6f05ef6ef56f6ef4ff5ee46f5ee3ff4ee37f4ed2ff4ec28f3ec21f3ec18f3eb11f3e90af2e704f1de01f0
d600eecf00edc600ecbe00ebb600ebaf00e9a800e9a100e69900e68f00e58800e48100e27800e27000e16900e06200e2
5c00e95800ee5300f44d00fa4800fb4200fb3c00fb3600fb2f00fb2b00fb2600fb2200fb2900fb3400fb3f00fb4300fb
4200fb4108fb3f1afb3e28fb3c33fb3b3efb3a4afb3954fb395efb386afb3775fb3780fc378afc3795fc379ffc38aafc
38b3fc39bdfc3ac8fc3acefc3ad0fc3ad2fc3ad6fc3ad8fc3adafc3addfc3ae0fc3ae1fc3be3fc3ae6fc3be9fc3bebfc
3bedfc3cf1fc3cf3fc3cf5fd3cf8fd3dfafd3dfbf93cf9ef3cf7ea3cf7e83cf6e73cf5e53cf5e33cf4e13bf3df3cf3de
3bf2dc3bf1db3bf1d93bf0d73befd53befd43beed13aeec73af0b73af1a93af29b39f28c3af48039f57239f56539f658
3af84b39f83e3af9333afa293af9243bf7233cf4223ef22140ef2043ed1f45ea1e47e81d4ae61c4de31b50e21b55df19
5ade1963e11a6ce21b76e41b7fe61d88e81e93ea1fa0ec20aaee21b4f022bef224c9f424d4f726def927e7fb29f3fc2b
fbfe2bfdfc2bfdf92bfdf72bfdf52bfdf22afdf02afced2afceb2afce929fce529fce329fce129fcdf28fcdc28fcda28
fcd828fcd527fbd127fbcb26fbc426fbbd25fbb825fbb124faab24faa524fa9f23fa9b23fa9422fa8f22fa8822fa8321
fa7d21f97820f97320f96d20f96620f9601ff95a1ff9551ff94f1ff94a1ff9451ef93f1ef93b1ef9361ef9321ef92d1e
f9291ef7261df4251df0251cec241ce9241be6231ae2221ae0221adc2119d92018d52018d21f17cf1e17cb1d16c81d16
c41c15c01c14bd1b14b91a13b51a13b31912af1812ac1711a81711a51710a1160f9e150f9b140e98130e94130d91120c
8d110c89110b87100b830f0a800f0a7d0e097a0d09750c08730b086f0b076d0a076909066608066207066006055c0605
""".split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", default="overthrust", help="directory of the inversion (default ./overthrust)")
    ap.add_argument("--band", default=BANDS[-1].name, choices=[b.name for b in BANDS],
                    help=f"plot the model after this band (default {BANDS[-1].name}, the final model)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator
    workdir = Path(args.workdir)
    hexs = "".join(_SWEEP_VP_HEX)
    cmap = ListedColormap([[int(hexs[i + k:i + k + 2], 16) / 255 for k in (0, 2, 4)]
                           for i in range(0, len(hexs), 6)], name="sweep_vp")
    true = load_true_model(workdir).astype(np.float64)
    start = starting_model(workdir).astype(np.float64)
    inv = np.load(workdir / f"inverted_vp_{args.band}.npy").astype(np.float64)
    assert inv.shape == NATIVE_SHAPE, inv.shape
    z = np.arange(NATIVE_SHAPE[0]) * NATIVE_DH
    lines = []
    for name, m in (("starting", start), ("inverted", inv)):
        e = m - true
        bands = ", ".join(f"{a / 1e3:g}-{b / 1e3:g} km {np.sqrt(np.mean(e[(z >= a) & (z < b)] ** 2)):.1f}"
                          for a, b in ((0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 4700)))
        lines.append(f"{name:9s} model{f' ({args.band})' if name == 'inverted' else ''}: r.m.s. error against the true model {np.sqrt(np.mean(e ** 2)):.1f} m/s "
                     f"({bands})")
    c = np.load(workdir / "curves.npz")
    lines.append(f"misfit mean(1-GCN): {c['loss'][0]:.4f} -> {c['loss'][-1]:.4f} over {len(c['loss'])} iterations; "
                 f"iterations took {c['iter_s'].sum() / 3600:.2f} h, peak {c['peak_gb'].max():.1f} GB")
    print("\n".join(lines))
    (workdir / "summary.txt").write_text("\n".join(lines) + "\n")

    # (1) sections and depth slice of the true, starting and inverted models
    ys, xs, zd = 11.375, 7.6, 1.5
    iy, ix, iz = (int(round(v * 1e3 / NATIVE_DH)) for v in (ys, xs, zd))
    models = (("True", true), ("Linear start", start), ("Inverted", inv))
    cut = {k: [m[:, iy, :], m[:, :, ix], m[iz]] for k, (_, m) in enumerate(models)}
    zmax, ymax, xmax = ((n - 1) * NATIVE_DH / 1e3 for n in NATIVE_SHAPE)
    ext = [[0, xmax, zmax, 0], [0, ymax, zmax, 0], [0, xmax, 0, ymax]]
    lbl = [("x, inline (km)", "depth (km)", f"inline section, y = {ys} km"),
           ("y, crossline (km)", "depth (km)", f"crossline section, x = {xs} km"),
           ("x, inline (km)", "y, crossline (km)", f"depth slice, z = {zd} km")]
    rng = []
    for fam in ((0, 1), (2,)):            # one colour scale for the sections, one for the slice
        lo, hi = np.percentile(np.concatenate([cut[k][r].ravel() for k in cut for r in fam]), [1, 99])
        for _ in fam:
            rng.append((np.floor(lo / 100) * 100, np.ceil(hi / 100) * 100))
    fig, axs = plt.subplots(3, 3, figsize=(20, 15.5), dpi=200)
    for r in range(3):
        for k, (name, _) in enumerate(models):
            kw = dict(aspect="auto", extent=ext[r], interpolation="nearest")
            if r == 2:
                kw["origin"] = "lower"
            axs[r, k].imshow(cut[k][r], cmap=cmap, vmin=rng[r][0], vmax=rng[r][1], **kw)
            axs[r, k].tick_params(labelsize=17)
            axs[r, k].set_xlabel(lbl[r][0], fontsize=20)
            if k == 0:
                axs[r, k].set_ylabel(lbl[r][1], fontsize=20)
            lab = f"({'abcdefghi'[r * 3 + k]})"
            axs[r, k].set_title(f"{lab} {name}" if r == 0 else lab, fontsize=24)
    fig.subplots_adjust(left=0.072, right=0.988, top=0.955, bottom=0.11, hspace=0.65, wspace=0.14)
    for r in range(3):
        pl, pr = axs[r, 0].get_position(), axs[r, -1].get_position()
        cb = fig.colorbar(axs[r, 0].images[0], cax=fig.add_axes([pl.x0, pl.y0 - 0.072, pr.x1 - pl.x0, 0.011]),
                          orientation="horizontal")
        cb.ax.tick_params(labelsize=15)
        cb.locator = MaxNLocator(nbins=9)
        cb.update_ticks()
        fig.text(0.018, 0.5 * (pl.y0 + pl.y1), lbl[r][2], rotation=90, va="center", ha="center", fontsize=19)
    fig.savefig(workdir / "overthrust_models.png", dpi=200)
    plt.close(fig)

    # (2) vertical profiles on the inline section
    fig, axs = plt.subplots(1, 3, figsize=(16, 9.5), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.09, wspace=0.08)
    for k, (ax, x) in enumerate(zip(axs, (4.0, 7.6, 14.0))):
        jx = int(round(x * 1e3 / NATIVE_DH))
        for lab, m, st in (("true model", true, dict(color="k", lw=2.6)),
                           ("starting model", start, dict(color="tab:blue", lw=2.2, ls="--")),
                           ("inverted model", inv, dict(color="tab:red", lw=2.2))):
            ax.plot(m[:, iy, jx], z / 1e3, label=lab, **st)
        ax.set_title(f"({'abc'[k]}) x = {x:.1f} km", fontsize=20)
        ax.set_xlim(2000, 6500)
        ax.set_xticks([2000, 3000, 4000, 5000, 6000])
        ax.set_xlabel("vp (m/s)", fontsize=18)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=15)
    axs[0].set_ylim(z[-1] / 1e3, 0)
    axs[0].set_ylabel("depth (km)", fontsize=18)
    axs[0].legend(fontsize=15, loc="lower left", frameon=False)
    fig.savefig(workdir / "overthrust_profiles.png", dpi=200)
    plt.close(fig)

    # (3) misfit across the cascade
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.0), dpi=200)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.13)
    ax.plot(np.arange(len(c["loss"])), c["loss"], color="#1f6f8b", lw=1.8)
    a = 0
    for i, band in enumerate(BANDS):
        b = a + band.epochs
        ax.axvspan(a - 0.5, b - 0.5, color="0.5", alpha=0.055 + 0.03 * (i % 2))
        if a:
            ax.axvline(a - 0.5, color="0.6", lw=0.9, ls=":")
        ax.annotate(f"2-{round(band.freqs[-1])} Hz", (0.5 * (a + b), 0.055), ha="center", fontsize=14.5,
                    color="0.3", fontweight="bold")
        a = b
    ax.set_xlabel("iteration (cascade cumulative)", fontsize=18)
    ax.set_ylabel("mean(1 $-$ GCN)", fontsize=18)
    ax.set_title("Misfit across the five bands", fontsize=18)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=15)
    ax.set_xlim(-2, TOTAL_EPOCHS + 1)
    ax.set_ylim(0.035, 0.80)
    fig.savefig(workdir / "overthrust_misfit.png", dpi=200)
    plt.close(fig)
    print("figures written to", workdir)


if __name__ == "__main__":
    main()
