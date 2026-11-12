#!/usr/bin/env python3
"""Closure tests and control plots for generator output.

  python validate.py out/tagged_10x110.npz

Checks: 4-momentum conservation, on-shell masses, and that reconstructing the collinear-frame
variables from the lab 4-vectors returns the thrown (x, Q2, alpha_p, pT, phi_p).
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from tagdis import kinematics as kin  # noqa: E402

d = np.load(sys.argv[1])
k, PD, kp, pp, X = d["k"], d["PD"], d["kp"], d["pp"], d["X"]

print("max |p_in - p_out|      :", np.abs(k + PD - kp - pp - X).max())
print("max |m_p(lab) - m_p|    :", np.abs(np.sqrt(kin.msq(pp)) - kin.M_P).max())
print("max |m_e(lab) - m_e|    :", np.abs(np.sqrt(np.clip(kin.msq(kp), 0, None)) - kin.M_E).max())
rec = kin.reconstruct(k, PD, kp, pp)
for key in ["x", "Q2", "alpha_p", "pT"]:
    print(f"max rel. diff {key:8s}:", np.max(np.abs(rec[key] - d[key]) / np.abs(d[key]).clip(1e-12)))
dphi = np.angle(np.exp(1j * (rec["phi_p"] - d["phi_p"])))
print("max |dphi_p|            :", np.abs(dphi[d["pT"] > 1e-6]).max())

w = d["weight"]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "lines.linewidth": 1.8})
STEP = dict(weights=w, histtype="step", color="C0")


def hist2d(a, x, y, bins, xlabel, ylabel, title):
    h = a.hist2d(x, y, bins=bins, weights=w, cmap="viridis", norm=LogNorm(), cmin=1e-30)
    fig.colorbar(h[3], ax=a, label="events")
    a.set(xlabel=xlabel, ylabel=ylabel, title=title)


def zoom(v, lo=0.1, hi=99.9):
    return np.percentile(v, [lo, hi])


fig, ax = plt.subplots(2, 3, figsize=(20, 11))

xb = np.logspace(*np.log10(zoom(d["x"], 0, 100)), 80)
qb = np.logspace(*np.log10(zoom(d["Q2"], 0, 100)), 80)
hist2d(ax[0, 0], d["x"], d["Q2"], [xb, qb], "x", r"$Q^2$ [GeV$^2$]", "Electron kinematics")
ax[0, 0].set(xscale="log", yscale="log")

ax[0, 1].hist(d["alpha_p"], bins=100, **STEP)
ax[0, 1].set(xlabel=r"$\alpha_p$", ylabel="events", yscale="log",
             title=r"Spectator light-cone fraction $\alpha_p$")

ax[0, 2].hist(1e3 * d["pT"], bins=100, **STEP)
ax[0, 2].set(xlabel=r"$p_T$ (collinear frame) [MeV]", ylabel="events", yscale="log",
             title=r"Spectator transverse momentum $p_T$")

ax[1, 0].hist(d["phi_p"], bins=36, range=(0, 2 * np.pi), **STEP)
ax[1, 0].set(xlabel=r"$\phi_p$ [rad]", ylabel="events", xlim=(0, 2 * np.pi), ylim=(0, None),
             title=r"Spectator azimuth $\phi_p$")

th = 1e3 * d["theta_p_beam"]
hist2d(ax[1, 1], d["p_lab"], th, [np.linspace(*zoom(d["p_lab"]), 100), np.linspace(0, zoom(th)[1], 100)],
       r"$p_{lab}$ proton [GeV]", r"$\theta$ w.r.t. hadron beam [mrad]", "Spectator proton in the lab")

ax[1, 2].hist(1e3 * rec["p_rest"], bins=100, **STEP)
ax[1, 2].set(xlabel=r"$|\vec p_R|$ (deuteron rest frame) [MeV]", ylabel="events", yscale="log",
             title="Spectator momentum in deuteron rest frame")

for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
out = sys.argv[1].replace(".npz", "_validation.png")
fig.savefig(out, dpi=200)
print("plots ->", out)
