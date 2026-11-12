# Tagged-DIS toy MC:  e + D → e' + X + p

A SIDIS-style event generator. The spectator proton is thrown in the photon–deuteron collinear frame,
and the generator builds the detector-frame 4-vectors event by event from the beam and electron 4-vectors.

```
python generate.py -n 100000 --ee 10 --pd 110 --unweighted -o out/tagged_10x110
python validate.py out/tagged_10x110.npz        # closure tests + control plots
```

Outputs: `*.hepmc` (HepMC3, ready for npsim/afterburner), `*.npz` (thrown variables, invariants,
4-vectors, weights) and `*_summary.json` (cross section and settings).

## Workflow
1. Throw `(x, Q², φ_e)` and `(α_p, p_T², φ_p)`. Samplers are in `tagdis/models.py`.
2. Build the scattered electron in the D rest frame, define the collinear frame (D at rest, **q along −z**,
   x-axis along the electron's transverse momentum), place the proton using `p⁺ = α_p M_D/2`,
   `p⁻ = (m_p² + p_T²)/p⁺`, and boost to the lab (`tagdis/kinematics.py`).
3. Weight: `w = σ_red / pdf_throw`, with
   `σ_red = dσ/(dx dQ² dφ_e dα_p dp_T² dφ_p)` in nb/GeV⁴. The weights sum to σ in the cuts.
4. Optionally unweight by accept–reject.

`kinematics.reconstruct()` is the inverse map from lab 4-vectors back to the collinear-frame variables.
The same function can be used on reconstructed (smeared) events.

## Conventions
- `x = Q²/(p_D·q)` (per nucleon, 0 < x < 2), `y = p_D·q / p_D·k`, `x̃ = x/(2 − α_p)`.
- `α_p = 2p_p⁺/p_D⁺`, `t = (p_D − p_p)²`.
- Beams: e along −z, D along +z. Use `--crossing-angle 0` together with the ePIC afterburner for
  the crossing angle and beam effects.
- Each event carries a random electron helicity `λ_e = ±1` and deuteron spin `S_D ∈ {−1, 0, 1}`
  (along the beam). They are passed to the cross section and written as HepMC attributes.
- HepMC: beams have status 4; e' and p have status 1; X is written as PDG 99 with status 2,
  so it is not tracked.

## Plugging in theory input
- **Cross section:** add a function `f(kin) -> array` to `CROSS_SECTIONS` in `tagdis/models.py`.
  `kin` contains the thrown variables, the invariants, the lab 4-vectors and the spins.
- **Throwing prescription:** pass any density `f(α, p_T)` to `GridSpectatorSampler`, or write a
  sampler that returns the variables together with their pdf.
- `ia-toy` is a **placeholder**: neutron DIS (toy F2) × Hulthén light-cone density, with no FSI or spin dependence.
