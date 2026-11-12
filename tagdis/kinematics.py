"""Lorentz kinematics for tagged DIS  e + D -> e' + X + p.

Frame conventions (Cosyn-Weiss / Strikman-Weiss style):

  * Collinear frame: deuteron at rest, virtual photon along -z.
    Light-cone components  p^{+-} = E +- p_z  (z = -q_hat).
    Spectator light-cone fraction  alpha_p = 2 p_p^+ / p_D^+ = 2 p_p^+ / M_D,  0 < alpha_p < 2.
  * x-axis = direction of the electron momentum transverse to q (lepton plane = x-z plane),
    y-axis = z_hat cross x_hat.  phi_p is the spectator azimuth about z in this basis.
  * x = Q^2 / (p_D . q)  (per-nucleon scaling variable, 0 < x < 2),
    y = (p_D . q) / (p_D . k).

All 4-vectors are numpy arrays with the last axis (E, px, py, pz) in GeV.
"""
import numpy as np

M_E = 0.51099895e-3
M_P = 0.93827208816
M_N = 0.93956542052
M_D = 1.87561294257
M_PI = 0.13957039


def mdot(a, b):
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def msq(a):
    return mdot(a, a)


def _unit(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def four(E, p3):
    return np.concatenate([np.asarray(E)[..., None], p3], axis=-1)


def boost(p, beta):
    """Boost 4-vectors p by velocity beta (same convention as ROOT TLorentzVector::Boost)."""
    p = np.asarray(p, dtype=float)
    beta = np.asarray(beta, dtype=float)
    shape = np.broadcast_shapes(p.shape, beta.shape[:-1] + (4,))
    p = np.broadcast_to(p, shape)
    beta = np.broadcast_to(beta, shape[:-1] + (3,))
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * p[..., 1:], axis=-1)
    g2 = np.where(b2 > 0, (gamma - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    E = gamma * (p[..., 0] + bp)
    p3 = p[..., 1:] + (g2 * bp + gamma * p[..., 0])[..., None] * beta
    return four(E, p3)


def beams(e_energy, d_momentum, crossing_angle=0.0):
    """Lab beam 4-vectors: electron along -z, deuteron along +z rotated by
    `crossing_angle` (rad) in the x-z plane (px = p sin(theta_c))."""
    pe = np.sqrt(e_energy**2 - M_E**2)
    k = np.array([e_energy, 0.0, 0.0, -pe])
    ED = np.hypot(d_momentum, M_D)
    PD = np.array([ED, d_momentum * np.sin(crossing_angle), 0.0, d_momentum * np.cos(crossing_angle)])
    return k, PD


def _rest_frame_electron(k, PD):
    beta_D = PD[1:] / PD[0]
    return boost(k, -beta_D), beta_D


def _collinear_axes(k_r, q_r):
    """Orthonormal basis (ex, ey, ez) of the collinear frame, expressed in the D rest frame."""
    ez = -_unit(q_r[..., 1:])
    k3 = np.broadcast_to(k_r[..., 1:], ez.shape)
    kperp = k3 - np.sum(k3 * ez, axis=-1, keepdims=True) * ez
    ex = _unit(kperp)
    ey = np.cross(ez, ex)
    return ex, ey, ez


def build_events(k, PD, x, Q2, phi_e, alpha, pT, phi_p):
    """Map thrown variables to lab-frame 4-vectors.

    Returns (valid, vecs) where vecs holds lab 4-vectors kp (scattered e), q, pp (spectator
    proton), X, and `valid` flags physically allowed events.
    """
    n = len(x)
    k_r, beta_D = _rest_frame_electron(k, PD)
    PDk = mdot(PD, k)
    y = Q2 / (x * PDk)
    nu = y * PDk / M_D

    E = k_r[0]
    kmag = np.linalg.norm(k_r[1:])
    Ep = E - nu
    valid = Ep > M_E
    kpmag = np.sqrt(np.clip(Ep**2 - M_E**2, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        cth = (2 * E * Ep - 2 * M_E**2 - Q2) / (2 * kmag * kpmag)
    valid &= np.abs(cth) <= 1.0
    cth = np.clip(np.nan_to_num(cth), -1.0, 1.0)
    sth = np.sqrt(1.0 - cth**2)

    e3 = k_r[1:] / kmag
    ref = np.array([0.0, 1.0, 0.0]) if abs(e3[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(ref, e3))
    e2 = np.cross(e3, e1)
    kp3 = kpmag[:, None] * (
        (sth * np.cos(phi_e))[:, None] * e1 + (sth * np.sin(phi_e))[:, None] * e2 + cth[:, None] * e3
    )
    kp_r = four(Ep, kp3)
    q_r = np.broadcast_to(k_r, (n, 4)) - kp_r

    ex, ey, ez = _collinear_axes(k_r, q_r)
    pplus = 0.5 * alpha * M_D
    pminus = (M_P**2 + pT**2) / pplus
    pz = 0.5 * (pplus - pminus)
    pp3 = (pT * np.cos(phi_p))[:, None] * ex + (pT * np.sin(phi_p))[:, None] * ey + pz[:, None] * ez
    pp_r = four(0.5 * (pplus + pminus), pp3)

    # X^+ in the collinear frame must be positive
    PD_r = np.array([M_D, 0.0, 0.0, 0.0])
    X_r = PD_r + q_r - pp_r
    valid &= (X_r[:, 0] + np.sum(X_r[:, 1:] * ez, axis=1)) > 0

    kp = boost(kp_r, beta_D)
    pp = boost(pp_r, beta_D)
    q = k - kp
    X = PD + q - pp
    return valid, {"kp": kp, "q": q, "pp": pp, "X": X}


def reconstruct(k, PD, kp, pp):
    """Inverse map: lab 4-vectors -> (x, Q2, y, alpha_p, pT, phi_p, ...).
    Usable on generated or reconstructed (smeared) events."""
    q = k - kp
    Q2 = -msq(q)
    PDq = mdot(PD, q)
    out = {"Q2": Q2, "x": Q2 / PDq, "y": PDq / mdot(PD, k)}
    beta_D = PD[..., 1:] / PD[..., :1]
    k_r = boost(k, -beta_D)
    q_r = boost(q, -beta_D)
    pp_r = boost(pp, -beta_D)
    ex, ey, ez = _collinear_axes(k_r, q_r)
    px, py, pz = (np.sum(pp_r[..., 1:] * e, axis=-1) for e in (ex, ey, ez))
    out["alpha_p"] = 2.0 * (pp_r[..., 0] + pz) / M_D
    out["pT"] = np.hypot(px, py)
    out["phi_p"] = np.mod(np.arctan2(py, px), 2 * np.pi)
    out["p_rest"] = np.linalg.norm(pp_r[..., 1:], axis=-1)
    return out


def derived(k, PD, vecs, alpha):
    """Invariants and lab observables saved with every event."""
    kp, q, pp, X = vecs["kp"], vecs["q"], vecs["pp"], vecs["X"]
    Q2 = -msq(q)
    x = Q2 / mdot(PD, q)
    t = msq(PD - pp)
    p_lab = np.linalg.norm(pp[:, 1:], axis=1)
    beam_axis = PD[1:] / np.linalg.norm(PD[1:])
    return {
        "W2": msq(PD + q),
        "MX2": msq(X),
        "t": t,
        "t_minus_mN2": t - M_N**2,
        "x_tilde": x / (2.0 - alpha),
        "p_lab": p_lab,
        "theta_p_lab": np.arccos(np.clip(pp[:, 3] / p_lab, -1, 1)),
        "theta_p_beam": np.arccos(np.clip(pp[:, 1:] @ beam_axis / p_lab, -1, 1)),
        "pL_over_pbeam": (pp[:, 1:] @ beam_axis) / (np.linalg.norm(PD[1:]) / 2.0),
        "E_e_lab": kp[:, 0],
        "theta_e_lab": np.arccos(np.clip(kp[:, 3] / np.linalg.norm(kp[:, 1:], axis=1), -1, 1)),
    }
