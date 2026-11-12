"""Physics plug-ins: how events are thrown, and the reduced cross section used to weight them.

This is the file the theory input goes into:
  * a sampler for the spectator variables (alpha_p, pT^2, phi_p)   -> SpectatorSampler
  * a reduced cross section  d sigma / (dx dQ2 dphi_e dalpha_p dpT^2 dphi_p)  [nb / GeV^4]
    as a function of the event kinematics                            -> CROSS_SECTIONS

A cross-section function receives `kin`, a dict of numpy arrays (thrown variables, invariants,
lab 4-vectors, beam helicities) and returns one value per event.  Anything not covered by
these placeholders (spin dependence, FSI, ...) only requires a new function here.
"""
import numpy as np

from .kinematics import M_N

ALPHA_EM = 1.0 / 137.035999
GEV2_TO_NB = 0.3893793721e6  # 1 GeV^-2 = 0.3894 mb


class ElectronSampler:
    """log-uniform in x, pdf ~ Q2^-q2_power in Q2 (1 = log-uniform, 2 follows 1/Q^4),
    uniform in phi_e."""

    def __init__(self, x_range, q2_range, q2_power=2.0):
        self.lx = np.log(x_range)
        self.q2_range = q2_range
        self.p = q2_power

    def _sample_q2(self, n, rng):
        lo, hi = self.q2_range
        if self.p == 1.0:
            q2 = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
            return q2, 1.0 / (np.log(hi / lo) * q2)
        e = 1.0 - self.p
        q2 = (lo**e + rng.uniform(0, 1, n) * (hi**e - lo**e)) ** (1.0 / e)
        return q2, e * q2**-self.p / (hi**e - lo**e)

    def sample(self, n, rng):
        x = np.exp(rng.uniform(*self.lx, n))
        Q2, pdf_q2 = self._sample_q2(n, rng)
        phi_e = rng.uniform(0, 2 * np.pi, n)
        # pdf in (x, Q2, phi_e)
        pdf = pdf_q2 / ((self.lx[1] - self.lx[0]) * 2 * np.pi * x)
        return {"x": x, "Q2": Q2, "phi_e": phi_e}, pdf


class SpectatorSampler:
    """Flat in alpha_p, pT^2 and phi_p (photon-deuteron collinear frame).
    Replace / subclass with an importance-sampling prescription; only `sample` must return
    the variables and their joint pdf in (alpha_p, pT^2, phi_p)."""

    def __init__(self, alpha_range, pt_max):
        self.alpha_range = alpha_range
        self.pt2_max = pt_max**2

    def sample(self, n, rng):
        alpha = rng.uniform(*self.alpha_range, n)
        pT = np.sqrt(rng.uniform(0, self.pt2_max, n))
        phi_p = rng.uniform(0, 2 * np.pi, n)
        vol = (self.alpha_range[1] - self.alpha_range[0]) * self.pt2_max * 2 * np.pi
        return {"alpha_p": alpha, "pT": pT, "phi_p": phi_p}, np.full(n, 1.0 / vol)


class GridSpectatorSampler(SpectatorSampler):
    """Importance sampling of (alpha_p, pT^2) from any density f(alpha, pT) tabulated on a grid
    (piecewise constant, so the pdf is exact); phi_p uniform.  A small flat admixture keeps the
    tails covered.  This is the hook for a theory-provided throwing prescription."""

    def __init__(self, alpha_range, pt_max, density, nbins=(200, 200), flat_frac=0.05):
        super().__init__(alpha_range, pt_max)
        self.ae = np.linspace(*alpha_range, nbins[0] + 1)
        self.te = np.linspace(0, self.pt2_max, nbins[1] + 1)
        ac, tc = 0.5 * (self.ae[1:] + self.ae[:-1]), 0.5 * (self.te[1:] + self.te[:-1])
        f = density(ac[:, None], np.sqrt(tc[None, :]))
        f = f / f.sum()
        f = (1 - flat_frac) * f + flat_frac / f.size
        self.cdf = np.cumsum(f.ravel())
        self.cell_pdf = f / (np.diff(self.ae)[:, None] * np.diff(self.te)[None, :])
        self.shape = f.shape

    def sample(self, n, rng):
        cell = np.minimum(np.searchsorted(self.cdf, rng.uniform(0, self.cdf[-1], n)), self.cdf.size - 1)
        ia, it = np.unravel_index(cell, self.shape)
        alpha = rng.uniform(self.ae[ia], self.ae[ia + 1])
        pT = np.sqrt(rng.uniform(self.te[it], self.te[it + 1]))
        phi_p = rng.uniform(0, 2 * np.pi, n)
        return {"alpha_p": alpha, "pT": pT, "phi_p": phi_p}, self.cell_pdf[ia, it] / (2 * np.pi)


# ---------------------------------------------------------------------------------------------
# Cross sections.  PLACEHOLDERS for testing the machinery -- to be replaced by the theory input.
# ---------------------------------------------------------------------------------------------

def xs_flat(kin):
    """Pure phase space: sum of weights = phase-space volume in the cuts."""
    return np.ones_like(kin["x"])


# Hulthen deuteron wave function (S-wave only), parameters in GeV
_HA, _HB = 0.2316 * 0.1973269804, 1.268 * 0.1973269804


def _psi2_unnorm(k):
    return (1.0 / (k**2 + _HA**2) - 1.0 / (k**2 + _HB**2)) ** 2


_kk = np.linspace(0, 5.0, 200001)
_HNORM = 1.0 / np.trapz(4 * np.pi * _kk**2 * _psi2_unnorm(_kk), _kk)


def lc_momentum_density(alpha, pT):
    """Light-cone momentum density rho(alpha_p, pT) with  int dalpha d^2pT rho = 1,
    rho = E_k |psi(k)|^2 / (alpha (2 - alpha)),  k^2 = (m^2 + pT^2)/(alpha (2 - alpha)) - m^2."""
    a2 = alpha * (2.0 - alpha)
    k2 = np.clip((M_N**2 + pT**2) / a2 - M_N**2, 0.0, None)
    return np.sqrt(M_N**2 + k2) * _HNORM * _psi2_unnorm(np.sqrt(k2)) / a2


def f2n_toy(x):
    return np.where(x < 1.0, 0.35 * x**-0.2 * (1.0 - x) ** 3, 0.0)


def xs_ia_toy(kin):
    """Toy impulse approximation: DIS on the neutron at x_tilde = x/(2 - alpha_p) times the
    Hulthen light-cone density.  Callan-Gross, no target-mass corrections, no FSI, unpolarized.
    Integrating over the spectator returns the neutron DIS cross section."""
    x, Q2, y, a, pT = kin["x"], kin["Q2"], kin["y"], kin["alpha_p"], kin["pT"]
    xt = x / (2.0 - a)
    dsig_n = 2 * np.pi * ALPHA_EM**2 / (xt * Q2**2) * (1.0 + (1.0 - y) ** 2) * f2n_toy(xt)
    # d^2pT = (1/2) dpT^2 dphi_p ;  dx_tilde/dx = 1/(2 - alpha) ;  phi_e integrates to 2 pi
    return GEV2_TO_NB * dsig_n / (2.0 - a) * 0.5 * lc_momentum_density(a, pT) / (2 * np.pi)


CROSS_SECTIONS = {"flat": xs_flat, "ia-toy": xs_ia_toy}


def make_spectator_sampler(name, alpha_range, pt_max):
    if name == "flat":
        return SpectatorSampler(alpha_range, pt_max)
    if name == "hulthen":
        return GridSpectatorSampler(alpha_range, pt_max, lc_momentum_density)
    raise ValueError(name)
