# Density-matrix locality in the water box

Computed with `trained_model/positive/mldftb_rewrite_initial.model`, matching the
original neutral/box spectrum analysis. This is not the separate energy/force-only
model. The same geometry-dependent Hamiltonian is used at both widths.
192 atoms, 960 orbitals, 128 electrons per spin, periodic cell approximately
13.734 Å on each side. The occupied/unoccupied gap is 0.124654 eV.

The spin-summed density is $P=U(f_\alpha+f_\beta)U^T$. Distances are minimum-image
atom distances, and each atom-pair block is 5 by 5. The relative truncation error
is $\sqrt{\sum_{d_{ij}>R}\|P_{ij}\|_F^2/\sum_{ij}\|P_{ij}\|_F^2}$.
`locality.png` shows radial block decay, total truncation error, and the largest
relative truncation error in any atom's block row.

| Radius (Å) | Mean retained atoms, including centre | Relative error, 0.025 eV | Relative error, 0.0001 eV |
|---:|---:|---:|---:|
| 3 | 8.3 | 15.73% | 15.96% |
| 4 | 21.2 | 5.52% | 5.63% |
| 5 | 40.8 | 2.83% | 2.91% |
| 6 | 68.3 | 1.38% | 1.42% |
| 6.8 | 98.7 | 0.684% | 0.705% |
| 8 | 146.5 | 0.220% | 0.224% |

The two densities have very similar spatial decay; lowering smearing removes
thermal occupation without improving locality. At 5 Å the worst atom-row error
is about 4.4%, and the largest discarded block norm is about 0.033. There is
substantial decay, but no sharp density cutoff at the Hamiltonian's 5 Å cutoff.

These are truncations of an already converged full density, not local-solver
predictions. On-site blocks and electron number are unchanged by this operation.
Band energy is also unchanged for radii at least 5 Å because H is zero outside
that range. This does not mean those remote density blocks can be discarded
during a solve without affecting local density, electrostatic energy, forces or
projector properties. In particular, positive semidefiniteness/idempotency are
not enforced by direct truncation.

The box is small: radii above roughly half the cell length (6.87 Å) are strongly
limited by periodic geometry, and this is the finite-cell Gamma-point density
matrix with periodic images combined. It is not an image-resolved infinite-system
density. Do not infer a bulk localization length or a 1500-atom sparsity advantage
from the far tail alone. A larger-cell calculation and a direct buffered/local
solve error test are required for that conclusion.

`radial.csv` and `truncation.csv` contain the numerical diagnostics.
`density_matrices.npz` contains H, both density matrices, pair distances and block
norms. `summary.json` records the model, geometry, widths, gap and populations.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/analyze_density_locality.py \
  --model examples/mldftb/trained_model/positive/mldftb_rewrite_initial.model \
  --xyz examples/mldftb/trained_model/box.xyz \
  --widths 0.025 0.0001 \
  --output examples/mldftb/trained_model/box/locality
```

## Band-energy truncation error

The signed total-system error is $Δ E_\mathrm{band}=(\gamma_R-\gamma):H$, with spin-summed $\gamma$.
This is saved as `band_energy_error_eV` in `truncation.csv` and shown in the fourth plot panel.

| Radius (Å) | Error at 0.025 eV (eV) | Error at 0.0001 eV (eV) |
|---:|---:|---:|
| 2 | 8.21899 | 8.07275 |
| 3 | 2.11494 | 2.07851 |
| 4 | 0.0311581 | 0.0307642 |
| 5 and above | 0 | 0 |

The contraction is exactly unchanged once all nonzero Hamiltonian blocks are retained.
