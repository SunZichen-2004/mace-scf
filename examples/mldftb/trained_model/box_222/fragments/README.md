# Fragment-size test in a 2×2×2 replicated box

Rebuilt the model Hamiltonian for 1536 atoms (7680 orbitals), using the same model and both smearing widths. Electron counts are 1024 per spin. Cell lengths are approximately 27.468 Å. The full-system gap is 0.124570 eV.

The Hamiltonian is saved in `../locality/hamiltonian.npy`; full density matrices and metadata are in that directory. Fragment results use a shared fitted chemical potential, conserving total electron number.

All 192 distinct core environments were solved using fragments of the larger Hamiltonian. Eightfold translational symmetry was checked: maximum Hamiltonian discrepancy 1.35e-9 eV (XYZ coordinate rounding), and repeated on-site reference blocks agree within 1e-8. Counts and energies carry multiplicity eight. This avoids eight redundant solves, not finite-size interactions. Timings are for the representative solves on one CPU thread, not a GPU or general 1536-atom performance benchmark.

| Radius (Å) | Mean fragment atoms | Density error, 0.025 eV | Density error, 0.0001 eV | Band error at 0.0001 eV (meV/atom) |
|---:|---:|---:|---:|---:|
| 3 | 8.3 | 0.044422 | 0.0943 | +6.1035 |
| 4 | 21.2 | 0.0060848 | 0.021288 | -0.52235 |
| 5 | 40.8 | 0.0016635 | 0.0036945 | -0.057817 |
| 6 | 68.3 | 0.00074053 | 0.0010226 | -0.00065605 |
| 6.8 | 98.7 | 0.00015458 | 0.00038509 | +0.00056745 |
| 8 | 160.7 | 2.1266e-05 | 3.6396e-05 | -0.00024935 |
| 10 | 309.7 | 1.1442e-06 | 4.9139e-06 | -1.5368e-06 |

The 5–6 Å accuracy is essentially unchanged: the favourable results there are not an artifact of fragments nearly filling the original cell. Above 6.8 Å, fragment membership differs substantially; the larger box continues to improve without nearing the full-system size. Band energy is a signed sum and need not converge monotonically.

All tested radii are below half the larger cell length. Larger buffers can still contain periodic-image interactions across their boundaries; this remains a finite periodic-cell principal-submatrix test. This is an exact replication rather than an independent larger water configuration.

The full-radius endpoint was omitted to avoid repeating an already validated exact-limit check on a 7680-dimensional matrix. The full reference was independently diagonalized, not copied from the smaller system.

Reproduce fragment analysis:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/analyze_local_fragments.py --input examples/mldftb/trained_model/box_222/locality --output examples/mldftb/trained_model/box_222/fragments --repeat-222-cores 192 --skip-full-radius
```
