# Overlapping fragment test

The full Hamiltonian is saved separately in `../locality/hamiltonian.npy` (960 by
960, float64, eV). Its atom order matches `../../box.xyz`, with five contiguous
orbitals per atom. Model and cell metadata are in `../locality/summary.json`.
The model is the same multipole-trained model used for the original box spectrum,
not the separate energy/force-only profiling model.

For each atom i, select all atoms within a minimum-image radius R, extract the
principal submatrix H_i and diagonalize it. Only the central atom's density block
and band-energy row are retained. The surrounding atoms act as a buffer; their
contributions are discarded, avoiding multiple counting across fragments.

Writing the fragment eigenpairs as (epsilon_ik, U_ik), the central-state weight is
$w_{ik}=\sum_{a\in i}|U_{ak}|^2$. For each spin, a common chemical potential is
fitted to $\sum_{ik}w_{ik}f(\epsilon_{ik},\mu_\sigma)=N_\sigma$.
The on-site block is formed from the central rows of U, and the band energy is
$\sum_{ik\sigma}w_{ik}\epsilon_{ik}f(\epsilon_{ik},\mu_\sigma)$.
This equals the sum of central-row gamma:H contractions within the fragments.
The chemical-potential search reuses the fragment eigensolutions.

Results also include a diagnostic using the exact full-system chemical potential
(`reference_mu`). The `fitted_mu` result does not use that reference and conserves
total electron number. Both widths reuse the same fragment diagonalizations.

For 0.0001 eV, with fitted chemical potential:

| Radius (Å) | Mean fragment atoms | Maximum on-site entry error | Maximum atom electron-count error | Band error (meV/atom) |
|---:|---:|---:|---:|---:|
| 4 | 21.2 | 0.0213 | 0.0400 | -0.523 |
| 5 | 40.8 | 0.00370 | 0.00531 | -0.0580 |
| 6 | 68.3 | 0.00102 | 0.00161 | -0.000643 |
| 6.8 | 98.7 | 0.000384 | 0.000614 | +0.000923 |

At 0.025 eV the maximum on-site entry errors for 5 and 6 Å are 0.00167 and
0.000738 respectively. The energy error can benefit from cancellation and is
not a bound on local density errors. No force or electrostatic-energy error is
measured here.

These are sequential one-thread CPU accuracy tests, not GPU performance results.
The full-radius endpoint shares one cached eigensolution across identical
fragments; its timing is therefore not comparable to solving distinct fragments.
The full-radius limit reproduces the exact densities and band energies within
numerical precision, providing an end-to-end check of the core weighting.

Fragments are induced submatrices of the finite periodic-cell Hamiltonian, not
explicit image-resolved open clusters. The cell is only 13.7 Å wide, so a larger
cell is needed to assess transferable buffer radii. The present core-weighted
construction is a density approximation; no claim of a global variational
energy/force formulation is made.

For a GPU prototype, 5–6 Å buffers are the most informative starting range.
Batching equal/padded-size fragments and grouping multiple core atoms per solve
could reduce launch overhead and repeated work. Neither improvement has been
benchmarked here.

Reproduce:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/analyze_local_fragments.py \
  --input examples/mldftb/trained_model/box/locality \
  --output examples/mldftb/trained_model/box/fragments
```
