import numpy as np
import bilby
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

cases = ['Case1', 'Case2', 'Case3', 'Case4']
true_dls = [150.0, 75.0, 100.0, 120.0]
true_thetas = [0.0, 1.0, 0.5, 1.57]

SIG_TH_BASE, SIG_D_BASE = 0.10, 0.058
ESS_MIN = 30
NSEED = 10          # set to 1 for a single-seed plot; use 10 to match Table 5

sig_th_grid = np.geomspace(0.03, 0.40, 20)
sig_d_grid = np.geomspace(0.01, 0.20, 20)     # fractional precision sigma_d/d_L


def kl_uniform(w):
    """KL(w || uniform) in nats; w must be normalized."""
    nz = w > 0
    return float(np.sum(w[nz] * np.log(len(w) * w[nz])))


def ess(w):
    return float(1.0 / np.sum(w ** 2))


def gauss_weights(x, center, sigma):
    """Normalized Gaussian likelihood weights, stabilized against underflow."""
    lw = -0.5 * ((x - center) / sigma) ** 2
    w = np.exp(lw - lw.max())
    return w / w.sum()


def grid_for_case(th_fold, dl, true_dl, true_th_fold, seeds):
    """
    Returns (Z, ESS_joint) on the grid, averaged over seeds.
      Z        = mean KL_sGRB - mean KL_HG  (the colored quantity)
      ESS_joint = mean joint ESS           (the hatching quantity)
    Each seed draws a fresh EM offset, matching how the pipeline works.
    """
    rng = np.random.default_rng(0)
    nS = len(seeds)
    # common random numbers across the grid, so cells are comparable
    z_th = rng.standard_normal(nS)
    z_d = rng.standard_normal(nS)

    Z = np.zeros((len(sig_d_grid), len(sig_th_grid)))
    E = np.zeros_like(Z)

    for row, sd in enumerate(sig_d_grid):
        for col, st in enumerate(sig_th_grid):
            kls, klh, ejs = [], [], []
            for s in range(nS):
                a = true_th_fold + st * z_th[s]           # simulated sGRB Theta measurement
                dm = true_dl + sd * true_dl * z_d[s]      # simulated HG distance measurement
                w_s = gauss_weights(th_fold, a, st)
                w_h = gauss_weights(dl, dm, sd * true_dl)
                w_j = w_s * w_h
                w_j /= w_j.sum()
                kls.append(kl_uniform(w_s))
                klh.append(kl_uniform(w_h))
                ejs.append(ess(w_j))
            Z[row, col] = np.mean(kls) - np.mean(klh)
            E[row, col] = np.mean(ejs)
    return Z, E


def draw_hatch(ax, mask, xe, ye):
    """Hatch exactly the masked cells (edges, not contour interpolation)."""
    for i, j in zip(*np.nonzero(mask)):
        ax.add_patch(Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i],
                               fill=False, hatch="////", edgecolor="k", lw=0))


def log_edges(c):
    c = np.asarray(c, float)
    mid = np.sqrt(c[1:] * c[:-1])
    return np.concatenate([[c[0] ** 2 / mid[0]], mid, [c[-1] ** 2 / mid[-1]]])


results = {}
for i, case in enumerate(cases):
    res = bilby.result.read_in_result(f"outdir/{case}_result.json")
    dl = res.posterior["luminosity_distance"].values
    th = res.posterior["theta_jn"].values
    th_fold = np.minimum(th, np.pi - th)
    true_th_fold = min(true_thetas[i], np.pi - true_thetas[i])
    Z, E = grid_for_case(th_fold, dl, true_dls[i], true_th_fold, range(NSEED))
    results[case] = (Z, E)

vmax = max(np.abs(Z).max() for Z, _ in results.values())
norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True, sharey=True, constrained_layout=True)
xe, ye = log_edges(sig_th_grid), log_edges(sig_d_grid * 100)
X, Y = np.meshgrid(sig_th_grid, sig_d_grid * 100)

for ax, case in zip(axes.flat, cases):
    Z, E = results[case]
    im = ax.pcolormesh(xe, ye, Z, cmap="RdBu_r", norm=norm, shading="flat", rasterized=True)
    ax.contour(X, Y, Z, levels=[0], colors="k", linewidths=1.4)
    draw_hatch(ax, E < ESS_MIN, xe, ye)
    ax.plot(SIG_TH_BASE, SIG_D_BASE * 100, marker="*", ms=12, mfc="gold", mec="k", mew=0.7, ls="none", zorder=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xticks([0.03, 0.05, 0.1, 0.2, 0.4]); ax.set_xticklabels(["0.03", "0.05", "0.1", "0.2", "0.4"])
    ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_title(case)

for ax in axes[1, :]:
    ax.set_xlabel(r"sGRB viewing-angle precision $\sigma_\theta$ (rad)")
for ax in axes[:, 0]:
    ax.set_ylabel(r"host-galaxy distance precision $\sigma_d/d_L$ (%)")

fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02,
             label=r"$\mathrm{KL_{sGRB}-KL_{HG}}$ (nats):  $+$ sGRB better,  $-$ HG better")
handles = [Line2D([], [], color="k", lw=1.4, label="crossover (tie)"),
           Line2D([], [], marker="*", ls="none", mfc="gold", mec="k", ms=9, label="this study's baseline"),
           Patch(fc="none", ec="k", hatch="////", label=f"joint ESS < {ESS_MIN}")]
fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)

fig.savefig("Figure4_Crossover.pdf")
fig.savefig("Figure4_Crossover.png", dpi=300)
print("saved Figure4_Crossover.pdf / .png")

# sanity check: baseline cell per case should match Table 5 joint ESS in sign and rough size
for case in cases:
    Z, E = results[case]
    bi = np.argmin(np.abs(sig_th_grid - SIG_TH_BASE))
    bj = np.argmin(np.abs(sig_d_grid - SIG_D_BASE))
    print(f"{case}: baseline cell  KL_sGRB-KL_HG = {Z[bj, bi]:+.2f}   joint ESS = {E[bj, bi]:.0f}")