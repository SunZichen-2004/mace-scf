# Energy/force-only model evaluation

Model: `../models/energy_force_only.model`. The neutral, positive and ideal-hydronium inputs are copies of the previous evaluation configurations; populations, geometry and electronic smearing were preserved.

Each case contains `configuration.xyz`, `orbitals.xyz` and `spectrum.png`. Every orbital frame includes unit-occupation electronic `density_coefficients` in the model channel order, with no nuclear contribution. `analysis/` contains frontier coefficient maps, geometry views, covariance checks and the ideal water/hydronium probes in `analysis.json`.

| Frontier gap (eV) | Previous model | Energy/force-only |
|---|---:|---:|
| Neutral H4O2, width 0.025 eV | 0.195579 | 0.308163 |
| Positive H11O5, width 0.0025 eV | 0.015246 | 0.038246 |
| Ideal hydronium, width 0.0025 eV | 0 (degenerate) | 0 (degenerate) |

The positive frontier states (indices 9 and 10) now have per-spin fillings 0.999524 and 0.000476, compared with 0.954747 and 0.045253 previously. They remain centred on the protonated-water group, with 99.1% and 97.9% of their squared coefficient weights there.

For exact ideal hydronium the lowest levels are -0.347028 eV (filled), -0.034737 eV (doublet, each half filled per spin), and 0.126925 eV (empty). The doublet again has C3 trace -1 and determinant +1. The total density retains threefold symmetry to 1.4e-15. A 1% stretch of one O-H bond splits the pair by 0.000274 eV, giving fillings 0.513708 and 0.486292 per spin. The tiny approximately 6e-11 eV splitting in the exported ideal configuration comes from finite XYZ coordinate precision; the analytic geometry probe is degenerate to floating-point precision.

Rotation and permutation Hamiltonian residuals are below 1.4e-15 eV. Saved eigenvalues reproduce exactly. Hamiltonians reconstructed from the rounded exported coefficient arrays agree within 7.1e-9 eV. Changing electron counts or smearing at fixed geometry leaves all eigenvalues unchanged.

The half-filled ideal-hydronium doublet therefore persists without multipole supervision. Multipole loss is not necessary for this ordering. This does not prove that it has no effect: the spectra and frontier localization differ between the two fitted models, and this is not a controlled ensemble of otherwise identical training runs. Symmetry requires the doublet's degeneracy but does not require this level ordering.

## Reproduce

From the repository root, for each of `neutral`, `positive`, and `ideal_hydronium`:

```bash
PYTHONPATH=. python scripts/export_mldftb_orbitals.py \
  --model examples/mldftb/trained_model/models/energy_force_only.model \
  --xyz examples/mldftb/trained_model/energy_force_only/neutral/configuration.xyz \
  --output examples/mldftb/trained_model/energy_force_only/neutral/orbitals.xyz \
  --plot examples/mldftb/trained_model/energy_force_only/neutral/spectrum.png
```

Then:

```bash
PYTHONPATH=. python scripts/analyze_mldftb_orbitals.py \
  --directory examples/mldftb/trained_model/energy_force_only \
  --output examples/mldftb/trained_model/energy_force_only/analysis \
  --model examples/mldftb/trained_model/models/energy_force_only.model \
  --ideal-water-probes
```
