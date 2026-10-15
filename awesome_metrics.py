import numpy as np, bilby
from em_sgrb_postprocessing import em_sgrb_postprocessing
from em_host_galaxy_postprocessing import em_host_galaxy_postprocessing

CASE, TRUE_DL, TRUE_THETA, NSEED = "Case4", 120.0, 1.57, 10

def wq(x, w, qs=(0.16, 0.50, 0.84)):                 # weighted quantiles (inverse CDF)
    i = np.argsort(x); cdf = np.cumsum(w[i]); cdf = cdf / cdf[-1]
    return np.interp(qs, cdf, x[i])

def summarize(dl, theta, th_fold, w):
    w = np.asarray(w, float); w = w / w.sum()
    q16, q50, q84 = wq(dl, w); t16, _, t84 = wq(th_fold, w)
    half = (q84 - q16) / 2
    return dict(width=half/q50, bias=(q50-TRUE_DL)/TRUE_DL, pull=(q50-TRUE_DL)/half,
                Theta_half=(t84-t16)/2, p_mirror=w[theta > np.pi/2].sum(), ess=1/np.sum(w**2))

res    = bilby.result.read_in_result(f"outdir/{CASE}_result.json")
dl0    = res.posterior["luminosity_distance"].values
theta0 = res.posterior["theta_jn"].values
fold0  = np.minimum(theta0, np.pi - theta0)
rows   = {"GW": [summarize(dl0, theta0, fold0, np.ones(len(dl0)))],
          "sGRB": [], "HG": [], "joint": []}

for seed in range(NSEED):
    theta, th_fold, w_s, *_ = em_sgrb_postprocessing(CASE, seed, TRUE_THETA)
    dl, w_h, *_             = em_host_galaxy_postprocessing(CASE, seed, TRUE_DL)
    assert np.allclose(dl, dl0) and np.allclose(theta, theta0)      # same samples, same order
    w_s, w_h = np.asarray(w_s, float), np.asarray(w_h, float)
    for name, w in [("sGRB", w_s), ("HG", w_h), ("joint", w_s*w_h)]:
        rows[name].append(summarize(dl, theta, th_fold, w))

for name, lst in rows.items():
    g = lambda k: np.array([r[k] for r in lst])
    print(f"{name:5s} width {g('width').mean():.3f}±{g('width').std():.3f} | "
          f"|bias| {abs(g('bias')).mean():.3f} | rms pull {np.sqrt((g('pull')**2).mean()):.2f} | "
          f"Theta_half {g('Theta_half').mean():.3f} | p_mirror {g('p_mirror').mean():.3f} | ESS {g('ess').mean():.0f}")