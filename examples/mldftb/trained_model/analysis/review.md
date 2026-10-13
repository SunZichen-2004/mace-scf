# Review of the exported orbitals

The covariance checks pass. The narrow charged frontier gap appears to be a property of the learned auxiliary Hamiltonian, rather than a rotation, ordering or eigensolver error. A symmetric hydronium probe exposes a partially occupied degenerate doublet, which is a useful warning about the learned level ordering and reduced electron count.

## Supplied configurations

| | Neutral | Positive |
|---|---:|---:|
| Composition | H4O2 | H11O5 |
| Charge | 0 | +1 |
| Alpha/beta populations | 4 / 4 | 10 / 10 |
| Smearing width (eV) | 0.025 | 0.0025 |
| Frontier indices (zero-based) | 3, 4 | 9, 10 |
| Frontier energies (eV) | -0.210564, -0.014985 | -0.138620, -0.123374 |
| Gap (eV) | 0.195579 | 0.015246 |
| Frontier filling, each spin | 0.974284, 0.014941 | 0.954747, 0.045253 |

These are different compositions and geometries, not a controlled charging comparison. The current non-SCF Hamiltonian depends on geometry and species, not electron counts or smearing. At fixed geometry, changing the alpha count by one changed all eigenvalues by exactly zero in both tests. Only occupations change. The difference between these two spectra therefore cannot be attributed directly to charging the same geometry.

The positive gap is about 6.1 times its smearing width, so the two states are primarily occupied and primarily empty, not half occupied. They are far from numerically degenerate in double precision. Increasing that configuration's width to 0.025 eV changes their per-spin fillings to approximately 0.6250 and 0.4753, without changing the spectrum. Other states also exchange occupation.

## Orbital shape and symmetry

Rotating positions and the cell, then transforming p coefficients as Cartesian vectors, reproduces the Hamiltonian within 1.5e-15 eV. Reversing atom order agrees within 1.3e-15 eV. Rotation-matched individual orbital overlaps are effectively one. Re-evaluation exactly reproduces the exported eigenvalues. These checks support the angular conventions and implementation, though they are not a proof for every configuration.

The neutral structure is a distorted hydrogen-bonded water dimer; donor and acceptor are different environments. The positive configuration is also asymmetric. Its central oxygen O4 has three nearby hydrogens H13, H14 and H15, with unequal O-H distances of approximately 0.987, 1.002 and 1.108 Angstrom. H15 is also only about 1.381 Angstrom from O1.

In the positive structure, orbital 9 has about 91.1% of its squared coefficient weight on O4/H13/H14/H15. Orbital 10 has about 85.8% there. The lower orbital favours H13/H14, whereas the higher orbital puts about 47.3% of its total weight on H15. This localization explains much of their visually different character. These weights describe the auxiliary orthonormal basis, not an integrated real-space charge partition.

See [frontier coefficients](positive_frontier.png) and [geometry with orbital weights and p vectors](positive_geometry.png). The geometry plot shows coefficient vectors, not real-space orbital isosurfaces. The two neutral plots provide the same diagnostics.

## Ideal symmetry probe

Using the same model, an isolated ideal C3v hydronium geometry (equal 0.98 Angstrom O-H bonds, hydrogens spaced by 120 degrees at height 0.35 Angstrom), charge +1, and N_alpha=N_beta=2 gives:

- Lowest level: -0.542244 eV, fully occupied in each spin.
- Next two levels: both -0.119051 eV, each occupied by 0.5 in each spin.
- Next level: -0.051293 eV, essentially empty at width 0.0025 eV.

The 120-degree rotation restricted to the frontier pair has trace -1 and determinant +1, identifying the two-dimensional E representation. The full one-spin density is invariant under the symmetry operation to 1.9e-15. Individual members of this degenerate pair need not be invariant: symmetry rotates them into each other, while their equally occupied sum is invariant. Stretching one O-H bond by 1% splits the pair by 0.000643 eV and changes their per-spin fillings to 0.5321 and 0.4679.

An ideal isolated water probe, also with four auxiliary electrons, instead has a gap of about 0.22064 eV. Together with the cluster localization, the hydronium result supports interpreting the charged frontier pair as a distorted/symmetry-split hydronium-like doublet. A continuous geometry path connecting the cluster and isolated molecule was not calculated, so that assignment remains an interpretation.

## Implications for the model

Multipole supervision constrains the mapped occupied density, not each individual orbital or all virtual eigenvalues. Energy/force supervision also constrains the Hamiltonian, but does not require its spectrum to reproduce a physical KS spectrum. The effective electron counts further change the filling problem. In the hydronium probe, four auxiliary electrons occupy one lower singlet plus half of a doublet; a closed-shell insulating auxiliary spectrum is therefore not obtained with this learned ordering.

This is not evidence of a numerical force-gradient failure. It is a potential modelling issue: a partially occupied doublet can create strong dependence on distortion and smearing. Whether that creates inaccurate charges, forces or an artificial structural instability requires comparing those observables along controlled distortions; the total energy also contains local and electrostatic terms that may compensate the band contribution. Raising smearing smooths occupations but does not change the level ordering or open a gap.

The next useful validation is a symmetric hydronium distortion scan at several widths, checking the total energy, forces and mapped multipoles against reference data. Before imposing a gap constraint, decide whether physical insulating orbital structure is a model requirement or whether accurate observables with an auxiliary spectrum are sufficient.

## Reproduction

From the repository root:

```bash
PYTHONPATH=. python scripts/analyze_mldftb_orbitals.py \
  --directory examples/mldftb/trained_model \
  --output examples/mldftb/trained_model/analysis \
  --model examples/mldftb/trained_model/positive/mldftb_rewrite_initial.model \
  --ideal-water-probes
```

Numerical results are in `analysis.json`. Original configurations, model and orbital exports were not modified.
