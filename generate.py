#!/usr/bin/env python3
"""Toy MC for tagged DIS  e + D -> e' + X + p  (spectator proton in the target fragmentation region).

Workflow (same as a SIDIS toy MC):
  1. throw electron variables (x, Q2, phi_e)  and spectator variables in the photon-deuteron
     collinear frame (alpha_p, pT, phi_p)                          -> tagdis/models.py samplers
  2. build lab 4-vectors from the beam 4-vectors, event by event   -> tagdis/kinematics.py
  3. weight with the reduced cross section                         -> tagdis/models.py
  4. write HepMC3 (for ePIC simulation) + .npz with all kinematics (for analysis)

Example:
  python generate.py -n 100000 --ee 10 --pd 110 --model ia-toy --unweighted -o out/tagged_10x110
"""
import argparse
import json
import os

import numpy as np

from tagdis import kinematics as kin
from tagdis.hepmc import HepMC3Writer
from tagdis.models import CROSS_SECTIONS, ElectronSampler, make_spectator_sampler


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--nevents", type=int, default=10000, help="events to write")
    ap.add_argument("-o", "--output", default="tagged_dis", help="output basename")
    ap.add_argument("--ee", type=float, default=10.0, help="electron beam energy [GeV]")
    ap.add_argument("--pd", type=float, default=110.0, help="deuteron momentum per nucleon [GeV/c]")
    ap.add_argument("--crossing-angle", type=float, default=0.0,
                    help="hadron beam angle in x-z plane [rad]. Keep 0 and use the ePIC afterburner "
                         "for crossing angle / beam effects (ePIC: -0.025)")
    ap.add_argument("--model", choices=sorted(CROSS_SECTIONS), default="ia-toy")
    ap.add_argument("--x", nargs=2, type=float, default=[1e-3, 0.9], metavar=("MIN", "MAX"))
    ap.add_argument("--q2", nargs=2, type=float, default=[2.0, 100.0], metavar=("MIN", "MAX"))
    ap.add_argument("--y", nargs=2, type=float, default=[0.01, 0.95], metavar=("MIN", "MAX"))
    ap.add_argument("--alpha", nargs=2, type=float, default=[0.8, 1.2], metavar=("MIN", "MAX"))
    ap.add_argument("--pt-max", type=float, default=0.4, help="max spectator pT [GeV]")
    ap.add_argument("--mx-min", type=float, default=2.0, help="min M_X [GeV]")
    ap.add_argument("--spectator-sampling", choices=["hulthen", "flat"], default="hulthen",
                    help="importance sampling of (alpha_p, pT^2)")
    ap.add_argument("--q2-power", type=float, default=2.0, help="throw Q2 with pdf ~ Q2^-power")
    ap.add_argument("--unweighted", action="store_true", help="accept-reject to unit weights")
    ap.add_argument("--nwarmup", type=int, default=500000, help="trials to find max weight")
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-hepmc", action="store_true")
    return ap.parse_args()


class Generator:
    def __init__(self, a):
        self.a = a
        self.k, self.PD = kin.beams(a.ee, 2.0 * a.pd, a.crossing_angle)
        self.esamp = ElectronSampler(a.x, a.q2, a.q2_power)
        self.ssamp = make_spectator_sampler(a.spectator_sampling, a.alpha, a.pt_max)
        self.xs = CROSS_SECTIONS[a.model]

    def trial(self, n, rng):
        """Throw n trial events. Returns (weights_per_trial_volume, event dict); weight=0 if cut."""
        ev, pdf_e = self.esamp.sample(n, rng)
        sp, pdf_s = self.ssamp.sample(n, rng)
        ev.update(sp)
        valid, vecs = kin.build_events(self.k, self.PD, ev["x"], ev["Q2"], ev["phi_e"],
                                       ev["alpha_p"], ev["pT"], ev["phi_p"])
        ev["y"] = ev["Q2"] / (ev["x"] * kin.mdot(self.PD, self.k))
        ev.update(kin.derived(self.k, self.PD, vecs, ev["alpha_p"]))
        ev["lambda_e"] = rng.choice([-1, 1], n)        # electron helicity
        ev["S_D"] = rng.choice([-1, 0, 1], n)          # deuteron spin projection on beam axis
        ev.update({"k": np.broadcast_to(self.k, (n, 4)), "PD": np.broadcast_to(self.PD, (n, 4))})
        ev.update(vecs)

        a = self.a
        valid &= (ev["y"] > a.y[0]) & (ev["y"] < a.y[1])
        valid &= ev["MX2"] > a.mx_min**2
        w = np.zeros(n)
        with np.errstate(all="ignore"):
            w[valid] = self.xs({key: val[valid] for key, val in ev.items()}) / (pdf_e[valid] * pdf_s[valid])
        w = np.nan_to_num(w, nan=0.0, posinf=0.0)
        return w, ev


SAVE_KEYS = ["x", "Q2", "y", "phi_e", "alpha_p", "pT", "phi_p", "W2", "MX2", "t", "t_minus_mN2",
             "x_tilde", "p_lab", "theta_p_lab", "theta_p_beam", "pL_over_pbeam", "E_e_lab",
             "theta_e_lab", "lambda_e", "S_D"]


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    gen = Generator(a)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)

    wmax = None
    if a.unweighted:
        w, _ = gen.trial(a.nwarmup, rng)
        wmax = 1.2 * w.max()
        if wmax <= 0:
            raise SystemExit("no events pass the cuts in warm-up")

    writer = None if a.no_hepmc else HepMC3Writer(a.output + ".hepmc")
    store = {key: [] for key in SAVE_KEYS + ["weight", "k", "PD", "kp", "pp", "X"]}
    ntrial, sumw, sumw2, nover, nwritten = 0, 0.0, 0.0, 0, 0
    masses = (kin.M_E, kin.M_D, kin.M_P)

    while nwritten < a.nevents:
        w, ev = gen.trial(a.batch, rng)
        ntrial += len(w)
        sumw += w.sum()
        sumw2 += (w**2).sum()
        if a.unweighted:
            nover += int((w > wmax).sum())
            keep = rng.uniform(0, wmax, len(w)) < w
        else:
            keep = w > 0
        idx = np.flatnonzero(keep)[: a.nevents - nwritten]
        for i in idx:
            wt = 1.0 if a.unweighted else w[i]
            for key in store:
                store[key].append(wt if key == "weight" else ev[key][i])
            if writer:
                mX = np.sqrt(max(ev["MX2"][i], 0.0))
                writer.write(nwritten, wt, ev["k"][i], ev["PD"][i], ev["kp"][i], ev["pp"][i],
                             ev["X"][i], masses + (mX,),
                             {"lambda_e": ev["lambda_e"][i], "S_D": ev["S_D"][i]})
            nwritten += 1
    if writer:
        writer.close()

    sigma = sumw / ntrial
    sigma_err = np.sqrt(max(sumw2 / ntrial - sigma**2, 0.0) / ntrial)
    data = {key: np.asarray(v) for key, v in store.items()}
    if not a.unweighted:
        data["weight"] = data["weight"] / ntrial      # weighted: sum(weight) = sigma [nb]
    np.savez_compressed(a.output + ".npz", **data)

    summary = {"sigma_nb": sigma, "sigma_err_nb": sigma_err, "n_trials": ntrial,
               "n_written": nwritten, "unweighted": a.unweighted, "n_overweight": nover,
               "beam_k": gen.k.tolist(), "beam_PD": gen.PD.tolist(),
               "note": "unweighted: each event carries sigma/n_written; weighted: npz weights sum to "
                       "sigma, HepMC W = weight*n_trials",
               "settings": vars(a)}
    with open(a.output + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"sigma = {sigma:.4e} +- {sigma_err:.1e} nb   ({nwritten} events, {ntrial} trials)")
    if nover:
        print(f"WARNING: {nover} trials exceeded w_max; increase --nwarmup or the safety factor")


if __name__ == "__main__":
    main()
